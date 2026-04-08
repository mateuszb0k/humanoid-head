import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import whisper
import speech_recognition as sr
import pyttsx3
from utils.weather import weather_prompt
import json
from utils.intent_module import IntentDetector
from gliner import GLiNER
import re
from utils.find_teacher import get_teacher_room
from utils.find_room import get_room_directions
import psutil
import os

GLINER_LABELS = ["room code", "person"]

def _ram_mb() -> float:
    """Zwraca aktualnie używany RAM przez ten proces w MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_ollama_ram_mb() -> float:
    """Znajduje proces ollama i zwraca jego zużycie RAM."""
    for proc in psutil.process_iter(['name']):
        try:
            if 'ollama' in proc.info['name'].lower():
                return proc.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return 0.0

def _tegrastats_ram_mb() -> tuple[float, float]:
    """
    Próbuje odczytać RAM z /proc/meminfo (całkowity system).
    Zwraca (used_MB, total_MB).
    Działa na Jetsonie i zwykłym Linuksie.
    """
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                key, val = line.split(":")[0], line.split(":")[1].strip().split()[0]
                info[key] = int(val)
        total = info["MemTotal"] / 1024
        available = info["MemAvailable"] / 1024
        used = total - available
        return used, total
    except Exception:
        return 0.0, 0.0

class RamMonitor:
    """
    Śledzi zużycie RAM per komponent.
    Użycie:
        mon = RamMonitor()
        mon.checkpoint("przed_whisper")
        model = whisper.load_model(...)
        mon.checkpoint("po_whisper")
        mon.report()
    """
    def __init__(self):
        self.snapshots: list[dict] = []

    def checkpoint(self, label: str):
        proc_mb = _ram_mb()
        sys_used, sys_total = _tegrastats_ram_mb()
        ollama_mb = get_ollama_ram_mb() 
        
        self.snapshots.append({
            "label": label,
            "proc_mb": proc_mb,
            "sys_used_mb": sys_used,
            "sys_total_mb": sys_total,
            "ollama_mb": ollama_mb 
        })
        print(f"[RAM] {label:30s} | proces: {proc_mb:7.1f} MB | system: {sys_used:7.1f} / {sys_total:.0f} MB | Ollama: {ollama_mb:7.1f} MB")

    def report(self):
        print("\n" + "═" * 70)
        print(f"{'Komponent':<25} {'Δ proces MB':>12} {'Δ system MB':>12} {'proces MB':>12}")
        print("═" * 70)

        pairs = [
            ("Python baseline",  "baseline",       "po_python"),
            ("Whisper STT",      "po_python",      "po_whisper"),
            ("LLM (Bielik)",     "po_whisper",     "po_llm"),
            ("GLiNER NER",       "po_llm",         "po_gliner"),
            ("IntentDetector",   "po_gliner",      "po_intent"),
        ]

        snap = {s["label"]: s for s in self.snapshots}

        for name, before_label, after_label in pairs:
            if before_label in snap and after_label in snap:
                b = snap[before_label]
                a = snap[after_label]
                delta_proc = a["proc_mb"] - b["proc_mb"]
                delta_sys  = a["sys_used_mb"] - b["sys_used_mb"]
                print(f"  {name:<23} {delta_proc:>+11.1f} {delta_sys:>+11.1f} {a['proc_mb']:>11.1f}")

        if self.snapshots:
            last = self.snapshots[-1]
            first = self.snapshots[0]
            print("─" * 70)
            print(f"  {'RAZEM (od startu)':<23} "
                  f"{last['proc_mb'] - first['proc_mb']:>+11.1f} "
                  f"{last['sys_used_mb'] - first['sys_used_mb']:>+11.1f} "
                  f"{last['proc_mb']:>11.1f}")
        print("═" * 70 + "\n")

    def tts_checkpoint(self, label: str = "TTS"):
        """Wywołaj PRZED i PO pyttsx3.init() żeby zmierzyć TTS."""
        self.checkpoint(label)


def preprocess_stt(text: str) -> str:
    text = re.sub(r'\b[nN]\s+[eE]\s*(\d+)', r'ne\1', text)
    text = re.sub(r'\b[eE]\s+[aA]\s*(\d+)', r'ea\1', text)
    text = re.sub(r'([a-zA-Z])\s*-\s*([a-zA-Z])', r'\1\2', text)
    text = re.sub(r'([a-zA-Z])\s*-\s*(\d)', r'\1\2', text)
    text = re.sub(r'([a-zA-Z])\s+(\d)', r'\1\2', text)
    text = re.sub(r'\b[nN](\d+)', r'ne\1', text)
    text = re.sub(r'\b[eE](\d+)', r'ea\1', text)
    return text

def split_building_numer(text: str) -> str:
    return re.sub(r'(?i)\b(ne|ea)(\d+)', r'\1,\2', text)


class NlpModel:
    def __init__(self, template=None):
        self.monitor = RamMonitor()

        self.monitor.checkpoint("baseline")

        self.model_stt = whisper.load_model("small")
        self.monitor.checkpoint("po_whisper")

        self.model_llm = OllamaLLM(
            model="mwiewior/bielik:7b-instruct-v0.1.Q3_K_M.gguf",
            temperature=0.1
        )
        self.monitor.checkpoint("po_llm")

        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        self.monitor.checkpoint("po_gliner")

        self.intent_detector = IntentDetector()
        self.monitor.checkpoint("po_intent")

        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        # Prompt 
        if template is None:
            template = "Tutaj jest pytanie do Ciebie: {question}"
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.model_llm

        self.monitor.report()

    def start(self):
        while True:
            question = input("Text: ")
            print("Analyzing...")

            intent = self.intent_detector.detect_intent(question)

            if intent == "POGODA":
                result = weather_prompt()
                self._tts_module("Poczekaj sprawdzam pogodę")
                self._tts_module("Szukam termometru")
                self._tts_module("Własnie dokonuje pomiaru")

            elif intent == "PG":
                data = preprocess_stt(question)
                entities = self.gliner_model.predict_entities(data, GLINER_LABELS, threshold=0.2)
                label_text = {e["label"]: e["text"] for e in entities}

                if entities:
                    if "room code" in label_text:
                        room = label_text["room code"].upper()
                        directions = get_room_directions(split_building_numer(room))
                        if directions.find("Błąd ") != -1:
                            result = directions.replace("Błąd ", "")
                        else:
                            result = f"Aby dojść do pokoju {room} {directions}"
                    elif "person" in label_text:
                        person = label_text["person"]
                        teacher_data = get_teacher_room(person)
                        if teacher_data["teacher_name"] is not None:
                            if teacher_data["room"] and teacher_data["building"]:
                                room = teacher_data["room"]
                                building = teacher_data["building"]
                                directions = get_room_directions(f"{building},{room}")
                                result = (f"{teacher_data['teacher_name']} jest w pokoju "
                                          f"{building}{room} aby dojść do {building}{room} {directions}")
                            else:
                                result = f"{teacher_data['teacher_name']} nie ma przypisanego pokoju"
                        else:
                            result = "Niestety nie zrozumiałem o kogo dokładnie Ci chodzi. Czy możesz powtórzyć?"
                else:
                    result = ("Jeśli chodzi o politechnikę Gdańską to jestem w stanie udzielać "
                              "informacji tylko o lokalizacji sal oraz wykładowców.")

            else:
                chunks = []
                iteration = 0
                for chunk in self.chain.stream({"question": question}):
                    text = chunk if isinstance(chunk, str) else str(chunk)
                    print(text, end="", flush=True)
                    chunks.append(text)
                    # check 1/5 tokens
                    iteration += 1
                    if iteration % 5 == 0:
                        proc_mb = _ram_mb()
                        sys_used, _ = _tegrastats_ram_mb()
                
                result = "".join(chunks)
                self.monitor.checkpoint("po_generowaniu_llm")

            print(f"\n[Odpowiedź]: {result}")
            # self._tts_module(result)

    def _stt_module(self):
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while True:
                print("Listening...")
                audio = self.recognizer.listen(source, phrase_time_limit=3)
                raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                raw_data = np.frombuffer(raw_data, dtype=np.int16)
                audio_np = raw_data.astype(np.float32) / 32768.0
                result = self.model_stt.transcribe(audio_np, fp16=False)
                speech = result["text"].strip()
                if speech:
                    return speech

    def _tts_module(self, text):
        """TTS z pomiarem RAM przed i po."""
        ram_before = _ram_mb()
        model_tts = pyttsx3.init()
        ram_after_init = _ram_mb()
        model_tts.say(text)
        model_tts.runAndWait()
        model_tts.stop()
        del model_tts
        ram_after_del = _ram_mb()
        print(f"[RAM][TTS] init: +{ram_after_init - ram_before:.1f} MB | "
              f"po del: {ram_after_del:.1f} MB (delta: {ram_after_del - ram_before:+.1f} MB)")


if __name__ == "__main__":
    template = f"""Jesteś wszechstronnym asystentem studentów Politechniki Gdańskiej. 

    Instrukcje zachowania:
    1. Odpowiadaj naturalnie i unikaj zbędnych powitań.
    Odpowiadasz krótko naturalnie i zwięźle. Masz absolutny zakaz używania emoji, gwiazdek, znaczników markdown 
    oraz wypunktowań — generuj wyłącznie czysty, spójny tekst mówiony.
    Odpowiadaj tylko i wyłącznie na temat.

    Pytanie od użytkownika:
    {{question}}
    """

    nlp = NlpModel(template=template)
    nlp.start()