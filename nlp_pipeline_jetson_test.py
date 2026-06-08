import queue
import threading
import re
import string
import time
import logging
from collections import deque
from queue import Queue
from threading import Thread

import numpy as np
import pyaudio
import speech_recognition as sr
import requests
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
from gliner import GLiNER
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from piper import SynthesisConfig
from piper.voice import PiperVoice
from rapidfuzz import process, fuzz

from utils.config import (
    TOP_3_THRESHOLD,
    GLINER_LABELS,
    MAP_NUMBERS,
    RANDOM_VOICE_LINES,
    THRESHOLD,
    CONVERSATION_BUFFER_LEN,
)
from utils.find_aud import get_aud_directions
from utils.find_room import get_room_directions
from utils.find_teacher import get_teacher_room, search_teacher
from utils.intent_module import IntentDetector
from utils.new_weather import weather_prompt


EMOTION_PL_MAP = {
    "Angry": "złość",
    "Disgust": "obrzydzenie",
    "Fear": "strach",
    "Happy": "radość",
    "Sad": "smutek",
    "Surprise": "zaskoczenie",
    "Neutral": "neutralność",
}


def preprocess_stt(text: str) -> str:
    text = re.sub(r"\b[nN]\s+[eE]\s*(\d+)", r"ne\1", text)
    text = re.sub(r"\b[eE]\s+[aA]\s*(\d+)", r"ea\1", text)
    text = re.sub(r"([a-zA-Z])\s*-\s*([a-zA-Z])", r"\1\2", text)
    text = re.sub(r"([a-zA-Z])\s*-\s*(\d)", r"\1\2", text)
    text = re.sub(r"([a-zA-Z])\s+(\d)", r"\1\2", text)
    text = re.sub(r"\b[nN](\d+)", r"ne\1", text)
    text = re.sub(r"\b[eE](\d+)", r"ea\1", text)
    return text


def split_building_numer(text: str) -> str:
    text = re.sub(r"(?i)\b(ne|ea)(\d+)", r"\1,\2", text)
    return text


class NlpModel:
    def __init__(self, template=None, using_mic=True, using_speaker=True):
        self.model_stt = WhisperModel(
            "base",
            device="cuda",
            compute_type="float16",
            num_workers=1,
        )

        model_path = "PiperTTS/pl_PL-mc_speech-medium.onnx"
        self.voice = PiperVoice.load(model_path)

        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source)

        self.intent_detector = IntentDetector()
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

        self.user_emotion = "Neutral"
        self.user_identity = "Unknown"
        self.active_identity = "Unknown"
        self._greeted_identity = None

        self.last_seen_at = time.time()
        self.pending_identity_change = False
        self.user_missing_timeout = 20.0

        self.llm_queue = deque()
        self.regex = re.compile(f"[{string.punctuation}]")
        self.result = ""
        self.end_of_result = False

        self.audio_queue = queue.Queue()
        self.last_mouth_update_at = 0.0
        self.mouth_update_interval = 0.1
        self.last_mouth_status_sent = 0.0

        self.playback_thread = threading.Thread(target=self.playback_handle, daemon=True)
        self.playback_thread.start()

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.voice.config.sample_rate,
            output=True,
            frames_per_buffer=4096,
        )

        self.model_llm = OllamaLLM(
            model="gemma4:e4b",
            temperature=0.4,
            reasoning=False,
        )

        self.using_mic = using_mic
        self.using_speaker = using_speaker
        self.tts_queue = Queue()

        if template is not None:
            self.prompt = ChatPromptTemplate.from_template(template)
        else:
            self.prompt = ChatPromptTemplate.from_template(
                "Tutaj jest pytanie do Ciebie: {question}"
            )

        self.chain = self.prompt | self.model_llm

    def start(self):
        self._stt_init()
        status = self._llm_init()
        print(f"LLM status: {status}")

        if not status:
            return

        while True:
            now = time.time()

            if self.user_identity in ("Unknown", None) and self.active_identity not in ("Unknown", None):
                if now - self.last_seen_at > self.user_missing_timeout:
                    self.active_identity = "Unknown"
                    self._greeted_identity = None
                    self.reset_context()

            if self.pending_identity_change:
                self.active_identity = self.user_identity
                self.pending_identity_change = False
                self.reset_context()

                if (
                    self.active_identity not in ("Unknown", "None", None)
                    and self.active_identity != self._greeted_identity
                ):
                    random_phrase = np.random.choice(
                        [
                            f"Cześć {self.active_identity}",
                            f"O cześć {self.active_identity}",
                            f"Miło znowu cię widzieć {self.active_identity}",
                        ]
                    )
                    self._tts_module(random_phrase)
                    self.audio_queue.join()
                    self._send_mouth_status_to_pi(0.0)
                    self.last_mouth_status_sent = 0.0
                    self._greeted_identity = self.active_identity

                continue

            if self.using_mic:
                question = self._stt_module()
                if question is None:
                    continue
            else:
                question = input("Text: ")

            self.get_new_user_name()

            self.end_of_result = False
            self.result = ""
            self.tts_queue = Queue()
            start_llm = 0
            end_llm = 0

            print("Analyzing...")

            intent = self.intent_detector.detect_intent(question)

            tts_thread = Thread(target=self._tts_stream)
            tts_thread.start()

            if intent == "POGODA":
                self.result = np.random.choice(RANDOM_VOICE_LINES) + weather_prompt()
                print(self.result)
                self.end_of_result = True

            elif intent == "AUD":
                print("Wykryto intencję: AUDYTORIUM")
                self.result = self.handle_specialrooms(question)
                print(self.result)
                self.end_of_result = True

            elif intent == "PG":
                data = preprocess_stt(question)
                entities = self.gliner_model.predict_entities(data, GLINER_LABELS, threshold=0.2)

                label_text = {}
                for entity in entities:
                    label_text[entity["label"]] = entity["text"]

                if entities:
                    if "room code" in label_text:
                        room = label_text["room code"].upper()
                        room_split = split_building_numer(room)
                        directions = get_room_directions(room_split)

                        if "Błąd " in directions:
                            self.result = directions.replace("Błąd ", "") + "."
                        else:
                            self.result = f"Aby dojść do pokoju {room} {directions}."

                    elif "person" in label_text:
                        person = label_text["person"]
                        top_3 = search_teacher(person)

                        if not top_3:
                            self.result = "Nie udało mi się znaleźć tej osoby w bazie."
                        else:
                            max_val = max(top_3.values())
                            best_teacher = None
                            candidates = list(top_3.keys())
                            iter_counter = 0

                            for k, v in top_3.items():
                                if v == max_val:
                                    best_teacher = k

                            if max_val > TOP_3_THRESHOLD:
                                teacher_data = get_teacher_room(best_teacher)
                                if teacher_data["teacher_name"] is not None:
                                    self.result = self.handle_teachers(teacher_data)
                                else:
                                    self.result = "Nie udało mi się znaleźć tej osoby w bazie."

                            elif max_val > THRESHOLD:
                                temp_result = "Nie jestem pewny o kogo ci chodzi..."
                                for n, k in enumerate(top_3.keys()):
                                    number = MAP_NUMBERS[str(n)]
                                    temp_result += f" Jeżeli chodzi ci o {k} powiedz {number}..."
                                print(temp_result)

                                if self.using_speaker:
                                    self._tts_module(temp_result)
                                    self.audio_queue.join()
                                    self._send_mouth_status_to_pi(0.0)
                                    self.last_mouth_status_sent = 0.0
                                else:
                                    print(temp_result)

                                while True:
                                    if self.using_mic:
                                        user_input = self._stt_module()
                                    else:
                                        user_input = input("podaj numer")

                                    if user_input is None:
                                        continue

                                    user_input = user_input.lower()

                                    if "zero" in user_input or user_input == "0":
                                        self.result = self.handle_teachers(get_teacher_room(candidates[0]))
                                        break
                                    elif "jeden" in user_input or user_input == "1":
                                        self.result = self.handle_teachers(get_teacher_room(candidates[1]))
                                        break
                                    elif "dwa" in user_input or user_input == "2":
                                        self.result = self.handle_teachers(get_teacher_room(candidates[2]))
                                        break
                                    else:
                                        if iter_counter < 2:
                                            if self.using_speaker:
                                                self._tts_module("Nie rozumiem powiedz jeszcze raz")
                                                self.audio_queue.join()
                                                self._send_mouth_status_to_pi(0.0)
                                                self.last_mouth_status_sent = 0.0
                                            else:
                                                print("Nie rozumiem powiedz jeszcze raz")
                                            iter_counter += 1
                                        else:
                                            if self.using_speaker:
                                                self._tts_module("Przepraszam nie jestem w stanie pomóc")
                                                self.audio_queue.join()
                                                self._send_mouth_status_to_pi(0.0)
                                                self.last_mouth_status_sent = 0.0
                                            else:
                                                print("Przepraszam nie jestem w stanie pomóc")
                                            self.result = "Przepraszam nie jestem w stanie pomóc."
                                            break
                            else:
                                self.result = (
                                    "Niestety nie zrozumiałem o kogo dokładnie Ci chodzi. "
                                    "Czy możesz powtórzyć swoje pytanie?"
                                )
                else:
                    self.result = (
                        "Jeśli chodzi o politechnikę Gdańską to jestem w stanie udzielać informacji "
                        "tylko o lokalizacji sal oraz wykładowców."
                    )

                self.end_of_result = True

            else:
                start_llm = time.time()
                end_llm = 0

                for chunk in self.chain.stream(
                    {
                        "question": question,
                        "history": self._get_history_buffer(),
                        "name": self.active_identity,
                        "emotion": self.user_emotion,
                    }
                ):
                    if not end_llm:
                        end_llm = time.time()

                    text = chunk if isinstance(chunk, str) else str(chunk)
                    self.result += text

                self.end_of_result = True

            tts_thread.join()
            self.audio_queue.join()
            self._send_mouth_status_to_pi(0.0)
            self.last_mouth_status_sent = 0.0
            time.sleep(0.3)

            self._handle_llm_queue(question, self.result)
            print(f"TTFT: {end_llm - start_llm}")

    def _stt_init(self):
        zeros = np.zeros(16000, dtype=np.float32)
        t0 = time.time()
        list(
            self.model_stt.transcribe(
                zeros,
                language="pl",
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
        )
        print(f"STT warm-up done in {time.time() - t0:.3f}s")

    def _handle_llm_queue(self, question, result):
        q = f"Użytkownik: {question}"
        a = f"LLM: {result}"

        if len(self.llm_queue) == 2 * CONVERSATION_BUFFER_LEN:
            self.llm_queue.popleft()
            self.llm_queue.popleft()

        self.llm_queue.append(q)
        self.llm_queue.append(a)

    def _get_history_buffer(self):
        result = ""
        for i, sentence in enumerate(self.llm_queue):
            result += str(i + 1) + f". {sentence} "
        return result

    def _tts_stream(self):
        if self.result == "":
            while self.result == "":
                time.sleep(0.1)

        text_said = ""
        max_chunk_without_punctuation = 120

        while True:
            current_text = str(self.result)
            remaining = current_text[len(text_said):]

            if current_text == text_said and self.end_of_result:
                break

            res = re.search(self.regex, remaining)
            if res is not None:
                cut = res.start() + 1
                text2say = remaining[:cut]
                if self.using_speaker:
                    self._tts_module(text2say)
                else:
                    print(text2say, end="", flush=True)
                text_said += text2say

            elif len(remaining) >= max_chunk_without_punctuation:
                text2say = remaining[:max_chunk_without_punctuation]
                if self.using_speaker:
                    self._tts_module(text2say)
                else:
                    print(text2say, end="", flush=True)
                text_said += text2say

            time.sleep(0.05)

    def _llm_init(self):
        try:
            _ = self.chain.invoke(
                {
                    "question": "Odpowiedz jednym słowem: OK",
                    "history": self._get_history_buffer(),
                    "name": "Unknown",
                    "emotion": "Neutralność",
                }
            )
            return True
        except Exception as e:
            print(e)
            return False

    def _stt_module(self):
        chunk_size = 1024
        rate = 16000
        silence_threshold = 0.01
        max_silence_chunks = 8

        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=chunk_size,
        )

        frames = []
        silent_chunks = 0
        recording = False

        print("Listening...")

        try:
            while True:
                data = stream.read(chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.float32)
                rms = np.sqrt(np.mean(chunk ** 2))

                if rms > silence_threshold:
                    recording = True
                    silent_chunks = 0
                    frames.append(chunk)
                elif recording:
                    frames.append(chunk)
                    silent_chunks += 1
                    if silent_chunks >= max_silence_chunks:
                        break

        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            return None

        audio_np = np.concatenate(frames)

        if len(audio_np) < 8000:
            return None

        start = time.time()
        segments, _ = self.model_stt.transcribe(
            audio_np,
            language="pl",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"[STT] time={time.time() - start:.3f}s text={text!r}")

        return text if text else None

    def _tts_module(self, text):
        phonemes = self.voice.phonemize(text)
        if len(phonemes):
            ids = list(self.voice.phonemes_to_ids(phonemes[0]))
            config = SynthesisConfig(length_scale=1.3)
            audio = self.voice.phoneme_ids_to_audio(ids, syn_config=config)

            sample_rate = self.voice.config.sample_rate
            chunk_size = max(1024, int(sample_rate * 0.050))

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                if len(chunk) == 0:
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                chunk_bytes = (chunk * 32767).astype(np.int16).tobytes()
                self.audio_queue.put((chunk_bytes, rms))

    def playback_handle(self):
        while True:
            audio_bytes, rms_volume = self.audio_queue.get()

            try:
                self.stream.write(audio_bytes)

                normalized_rms = max(0.0, min(rms_volume / 0.095, 1.0))
                now = time.time()

                should_send = (
                    now - self.last_mouth_update_at >= self.mouth_update_interval
                    or abs(normalized_rms - self.last_mouth_status_sent) >= 0.15
                )

                if should_send:
                    self._send_mouth_status_to_pi(normalized_rms)
                    self.last_mouth_update_at = now
                    self.last_mouth_status_sent = normalized_rms

            finally:
                self.audio_queue.task_done()

    def handle_teachers(self, teacher_data):
        if teacher_data["room"] is not None and teacher_data["building"] is not None:
            room = teacher_data["room"]
            building = teacher_data["building"]
            room_directions = get_room_directions(f"{building},{room}")
            result = (
                f"{teacher_data['teacher_name']} jest w pokoju {building}{room} "
                f"aby dojść do {building}{room} {room_directions}."
            )
        else:
            result = f"{teacher_data['teacher_name']} nie ma przypisanego pokoju."
        return result

    def handle_specialrooms(self, question: str):
        q = question.lower().replace(".", "").replace(",", "")
        building = "NE"

        if "ea" in q or "star" in q:
            building = "EA"

        if "bibliotek" in q or "czytelni" in q:
            target = "NE, BIBLIOTEKA"
        elif "stołów" in q or "jedzen" in q or "bar" in q or "jadalni" in q:
            target = "NE, STOŁÓWKA"
        elif "szatni" in q:
            target = f"{building}, SZATNIA"
        elif "2" in q or "dwa" in q or "drug" in q:
            target = f"{building}, AUD2"
        elif "lew" in q:
            target = "NE, AUD1 LEWE"
        elif "praw" in q:
            target = "NE, AUD1 PRAWE"
        elif "1" in q or "jeden" in q or "pierwsz" in q:
            if building == "EA":
                target = "EA, AUD1"
            else:
                return "W Nowym ETI audytorium pierwsze dzieli się na lewe i prawe. Sprecyzuj proszę, o które ci chodzi."
        else:
            return "O które audytorium lub miejsce dokładnie pytasz?"

        directions = get_aud_directions(target)
        if "Błąd" in directions or "Error" in directions:
            return "Wybacz, nie potrafię odnaleźć drogi do tego miejsca."

        return directions

    def get_new_user_name(self):
        if self.user_identity != "Unknown":
            return

        self._tts_module("Hej, chyba się jeszcze nie znamy.")
        self._tts_module("Powiedz mi swoje imię, abym mógł cię zapamiętać.")
        self.audio_queue.join()
        self._send_mouth_status_to_pi(0.0)
        self.last_mouth_status_sent = 0.0
        time.sleep(0.3)

        name_file = "imiona_polskie.txt"
        names = []
        with open(name_file, encoding="utf-8") as f:
            for line in f:
                names.append(line.strip())

        new_name_raw = None

        if self.using_mic:
            while not new_name_raw:
                if self.user_identity != "Unknown":
                    return
                new_name_raw = self._stt_module()
        else:
            new_name_raw = input("Podaj imię: ")

        if not new_name_raw:
            return

        new_name_phrase = new_name_raw.replace(".", "").replace(",", "").strip()
        new_name = None
        new_name_score = 0

        for w in new_name_phrase.split():
            match = process.extractOne(w, names, scorer=fuzz.WRatio)
            if match and match[1] > new_name_score:
                new_name_score = match[1]
                new_name = match[0]

        self.user_identity = new_name if new_name is not None else "Unknown"
        self.active_identity = self.user_identity
        self._greeted_identity = self.user_identity

        self._tts_module(f"Miło cię poznać {self.user_identity}")
        self.audio_queue.join()
        self._send_mouth_status_to_pi(0.0)
        self.last_mouth_status_sent = 0.0

        td = Thread(target=self._save_name_to_pi, daemon=True)
        td.start()

    def _save_name_to_pi(self):
        try:
            rpi_url = "http://uncanny-head.local:5000/api/save_name"
            requests.post(rpi_url, json={"identity": self.user_identity}, timeout=0.5)
        except Exception as e:
            print(e)

    def _send_mouth_status_to_pi(self, status):
        # try:
        #     rpi_url = "http://uncanny-head.local:5000/api/mouth_status"
        #     requests.post(rpi_url, json={"mouth_status": str(status)}, timeout=0.2)
        # except Exception as e:
        #     print(e)
        return

    def reset_context(self):
        self.llm_queue = deque()
        self.result = ""
        self.end_of_result = False
        self.tts_queue = Queue()


log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)
app = Flask(__name__)


@app.route("/api/data", methods=["POST"])
def handle_data():
    data = request.json or {}
    incoming_identity = data.get("identity", "Unknown")
    incoming_emotion = data.get("emotion", "Neutral")

    now = time.time()
    nlp.last_seen_at = now

    normalized_identity = (
        incoming_identity if incoming_identity not in (None, "", "None") else "Unknown"
    )

    previous_identity = nlp.user_identity

    if (
        normalized_identity not in ("Unknown",)
        and previous_identity not in ("Unknown", None)
        and normalized_identity != previous_identity
    ):
        nlp.user_identity = normalized_identity
        nlp.pending_identity_change = True

    elif previous_identity in ("Unknown", None) and normalized_identity not in ("Unknown",):
        nlp.user_identity = normalized_identity
        nlp.active_identity = normalized_identity

    elif normalized_identity == "Unknown":
        nlp.user_identity = "Unknown"

    nlp.user_emotion = EMOTION_PL_MAP.get(
        str(incoming_emotion).capitalize(),
        "neutralność",
    )

    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    template = """Jesteś asystentem głosowym dla studentów Politechniki Gdańskiej na wydziale Elektroniki Telekomunikacji i Informatyki.

Mów krótko, naturalnie i konkretnie. Brzmij jak pomocny rozmówca, a nie jak system komunikatów.

Masz dostęp do:
- imienia użytkownika: {name}
- aktualnie wykrytej emocji użytkownika: {emotion}
- ostatniej historii rozmowy: {history}

Zasady:
- Najpierw odpowiedz sensownie na pytanie użytkownika.
- Możesz użyć imienia użytkownika, ale tylko jeśli brzmi to naturalnie. Nie musisz używać go w każdej odpowiedzi.
- Możesz krótko odnieść się do emocji użytkownika, ale tylko wtedy, gdy to naprawdę pomaga w rozmowie. Nie zaczynaj każdej odpowiedzi od komentarza o emocji.
- Jeśli emocja to neutralność, zwykle nie komentuj jej wprost.
- Jeśli pytanie jest konkretne, odpowiedz konkretnie.
- Jeśli nie wiesz, powiedz to wprost.
- Nie wymyślaj informacji o planie zajęć ani danych, których nie masz.
- Odpowiedzi mają być dobre do wypowiedzenia przez TTS, więc używaj prostych zdań i naturalnej interpunkcji.
- Nie używaj markdownu, emoji, list punktowanych ani kodu.
- Unikaj zbędnych wstępów i powitań.

Kontekst rozmowy:
{history}

Pytanie użytkownika:
{question}
"""

    nlp = NlpModel(template=template, using_mic=True, using_speaker=True)
    td = Thread(
        target=lambda: app.run("0.0.0.0", 5000, debug=False, use_reloader=False),
        daemon=True,
    )
    td.start()
    nlp.start()