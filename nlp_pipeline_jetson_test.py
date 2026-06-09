import queue
import sys
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
    text = re.sub(r'\b([nNeE])\.\s*([eEaA])\.\s*(\d+)\b', lambda m: (m.group(1) + m.group(2) + m.group(3)).lower(), text)
    text = re.sub(r'\b([nNeE])\.\s*([eEaA])\s*(\d+)\b', lambda m: (m.group(1) + m.group(2) + m.group(3)).lower(), text)
    text = re.sub(r'\b[nN]\s+[eE]\s*(\d+)', r'ne\1', text)
    text = re.sub(r'\b[eE]\s+[aA]\s*(\d+)', r'ea\1', text)
    text = re.sub(r'([a-zA-Z])\s*-\s*([a-zA-Z])', r'\1\2', text)
    text = re.sub(r'([a-zA-Z])\s*-\s*(\d)', r'\1\2', text)
    text = re.sub(r'([a-zA-Z])\s+(\d)', r'\1\2', text)
    text = re.sub(r'\b[nN](\d+)', r'ne\1', text)
    text = re.sub(r'\b[eE](\d+)', r'ea\1', text)
    return text


def split_building_numer(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r'[\.\s]+', '', text)
    text = re.sub(r'(?i)\b(NE|EA)(\d+)\b', r'\1,\2', text)
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
        self.face_visible = False
        self.was_face_visible = False

        self.llm_queue = deque()
        self.regex = re.compile(f"[{string.punctuation}]")
        self.result = ""
        self.end_of_result = False

        self.audio_queue = queue.Queue()
        self.last_mouth_update_at = 0.0
        self.mouth_update_interval = 0.1
        self.last_mouth_status_sent = 0.0
        self.is_speaking = False
        self.post_tts_cooldown = 0.6
        self.last_tts_finished_at = 0.0

        self.playback_thread = threading.Thread(target=self.playback_handle, daemon=True)
        self.playback_thread.start()

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.voice.config.sample_rate,
            output=True,
            frames_per_buffer=8192,
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

            if self.active_identity not in ("Unknown", None):
                if now - self.last_seen_at > self.user_missing_timeout:
                    self.active_identity = "Unknown"
                    self.user_identity = "Unknown"
                    self._greeted_identity = None
                    self.reset_context()

            if not self.face_visible:
                if self.was_face_visible:
                    self.stop_context()
                    self.was_face_visible = False
                time.sleep(0.1)
                continue

            if not self.was_face_visible:
                self.was_face_visible = True
                time.sleep(1.5)
                self.pending_identity_change = True

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
                    self.say(random_phrase)
                    self._greeted_identity = self.active_identity

                continue

            if self.using_mic:
                question = self._stt_module()
                if question is None:
                    continue
            else:
                question = input("Text: ")

            if self.user_identity == "Unknown":
                self.get_new_user_name()
                continue

            self.end_of_result = False
            self.result = ""
            self.tts_queue = Queue()
            start_llm = 0
            end_llm = 0
            tts_thread = None
            print(f"[GREETING] pending={self.pending_identity_change} active={self.active_identity} greeted={self._greeted_identity}")
            print("Analyzing...")

            intent = self.intent_detector.detect_intent(question)

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
                                prompt_lines = ["Nie jestem pewny o kogo ci chodzi."]
                                for n, k in enumerate(top_3.keys()):
                                    number = MAP_NUMBERS[str(n)]
                                    prompt_lines.append(f"Jeżeli chodzi ci o {k}, powiedz {number}.")
                                prompt_lines.append("Jeśli to nie jest żadna z tych osób, powiedz pięć.")

                                temp_result = " ".join(prompt_lines)
                                print(temp_result)

                                if self.using_speaker:
                                    for line in prompt_lines:
                                        self.say(line)
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
                                    elif "pięć" in user_input or user_input == "5":
                                        self.result = "Przepraszam, tym razem nie byłem w stanie pomóc. Spróbuj proszę zapytać jeszcze raz."
                                        break
                                    else:
                                        if iter_counter < 2:
                                            if self.using_speaker:
                                                self.say("Nie rozumiem powiedz jeszcze raz")
                                            else:
                                                print("Nie rozumiem powiedz jeszcze raz")
                                            iter_counter += 1
                                        else:
                                            if self.using_speaker:
                                                self.say("Przepraszam nie jestem w stanie pomóc")
                                            else:
                                                print("Przepraszam nie jestem w stanie pomóc")
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
                self.is_speaking = True
                tts_thread = Thread(target=self._tts_stream)
                tts_thread.start()
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

            if tts_thread is not None:
                tts_thread.join()
                self._finish_speaking()
            elif self.result and self.using_speaker:
                self.say(self.result)
            elif self.result:
                print(self.result)

            time.sleep(0.2)

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
        while self.result == "" and not self.end_of_result:
            time.sleep(0.03)

        text_said = ""
        min_chunk_chars = 80
        max_chunk_chars = 180
        sentence_end_regex = re.compile(r"[.!?]")

        while True:
            current_text = str(self.result)
            remaining = current_text[len(text_said):]

            if not remaining and self.end_of_result:
                break

            if not remaining:
                time.sleep(0.03)
                continue

            chunk = None

            if len(remaining) >= min_chunk_chars:
                matches = list(sentence_end_regex.finditer(remaining))
                if matches:
                    cut = None
                    for match in matches:
                        if match.end() >= min_chunk_chars:
                            cut = match.end()
                            break

                    if cut is None:
                        cut = matches[-1].end()

                    chunk = remaining[:cut]

            if chunk is None and len(remaining) >= max_chunk_chars:
                cut = remaining.rfind(" ", 0, max_chunk_chars)
                if cut == -1:
                    cut = max_chunk_chars
                chunk = remaining[:cut]
            if chunk is None and self.end_of_result:
                chunk = remaining

            if chunk:
                if self.using_speaker:
                    self._tts_module(chunk)
                else:
                    print(chunk, end="", flush=True)
                text_said += chunk

            time.sleep(0.03)

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
        while self.is_speaking or time.time() - self.last_tts_finished_at < self.post_tts_cooldown:
            time.sleep(0.05)

        chunk_size = 1024
        rate = 16000

        speech_start_threshold = 0.015
        speech_end_threshold = 0.008
        max_silence_chunks = 8
        max_record_seconds = 6
        max_chunks = int(rate * max_record_seconds / chunk_size)

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
        total_chunks = 0

        print("Listening...")

        try:
            while True:
                if not self.face_visible:
                    frames=[]
                    break

                data = stream.read(chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.float32)
                rms = np.sqrt(np.mean(chunk ** 2))
                total_chunks += 1

                if not recording:
                    if rms > speech_start_threshold:
                        recording = True
                        frames.append(chunk)
                else:
                    frames.append(chunk)

                    if rms < speech_end_threshold:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0

                    if silent_chunks >= max_silence_chunks:
                        break

                    if total_chunks >= max_chunks:
                        print("Max speech length reached")
                        break

        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            return None

        audio_np = np.concatenate(frames)

        if len(audio_np) < 8000:
            return None

        segments, _ = self.model_stt.transcribe(
            audio_np,
            language="pl",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text if text else None

    def _tts_module(self, text):
        chunk_size = 8192
        phonemes = self.voice.phonemize(text)
        if len(phonemes):
            ids = list(self.voice.phonemes_to_ids(phonemes[0]))
            config = SynthesisConfig(length_scale=1.3)
            audio = self.voice.phoneme_ids_to_audio(ids, syn_config=config)

            sample_rate = self.voice.config.sample_rate
            # chunk_size = max(1024, int(sample_rate * 0.035))


            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                if len(chunk) == 0:
                    continue

                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

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

        self.say("Hej, chyba się jeszcze nie znamy.")
        self.say("Powiedz mi swoje imię, abym mógł cię zapamiętać.")
        time.sleep(0.3)

        name_file = "imiona_polskie.txt"
        names = []
        with open(name_file, encoding="utf-8") as f:
            for line in f:
                names.append(line.strip())

        new_name_raw = None

        if self.using_mic:
            while not new_name_raw:
                if self.user_identity != "Unknown" or not self.face_visible:
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

        self.say(f"Miło cię poznać {self.user_identity}")

        td = Thread(target=self._save_name_to_pi, daemon=True)
        td.start()

    def _save_name_to_pi(self):
        try:
            rpi_url = "http://uncanny-head.local:5000/api/save_name"
            requests.post(rpi_url, json={"identity": self.user_identity}, timeout=0.5)
        except Exception as e:
            print(e)

    def _send_mouth_status_to_pi(self, status):
        try:
            rpi_url = "http://uncanny-head.local:5000/api/mouth_status"
            requests.post(rpi_url, json={"mouth_status": str(status)}, timeout=0.2)
        except Exception as e:
            print(e)
        return

    def reset_context(self):
        self.llm_queue = deque()
        self.result = ""
        self.end_of_result = False
        self.tts_queue = Queue()

    def stop_context(self):
        self.end_of_result = True
        self.result=""
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.is_speaking = False


    def say(self, text: str, wait: bool = True):
        if not text:
            return

        self.is_speaking = True
        self._tts_module(text)

        if wait:
            self._finish_speaking()

    def _finish_speaking(self):
        self.audio_queue.join()
        self._send_mouth_status_to_pi(0.0)
        self.last_mouth_status_sent = 0.0
        self.last_tts_finished_at = time.time()
        time.sleep(self.post_tts_cooldown)
        self.is_speaking = False


log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)
app = Flask(__name__)


@app.route("/api/data", methods=["POST"])
def handle_data():
    data = request.json or {}
    face_visible = data.get("face_visible",False)
    nlp.face_visible = face_visible

    incoming_identity = data.get("identity", "Unknown")
    incoming_emotion = data.get("emotion", "Neutral")

    if face_visible:
        nlp.last_seen_at = time.time()

    raw_identity = str(incoming_identity).strip()
    if raw_identity.lower() in ("", "none", "unknown", "null"):
        normalized_identity = "Unknown"
    else:
        normalized_identity = raw_identity

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

        nlp.pending_identity_change = True

    elif normalized_identity == "Unknown":
        nlp.user_identity = "Unknown"

    nlp.user_emotion = EMOTION_PL_MAP.get(
        str(incoming_emotion).capitalize(),
        "neutralność",
    )
    print(f"[API] incoming_identity={incoming_identity} normalized_identity={normalized_identity} previous={previous_identity}")
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    arg_mic = sys.argv[1] == "True" if len(sys.argv) > 1 else True
    arg_speaker = sys.argv[2] == "True" if len(sys.argv) > 2 else True
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
- Możesz czasami wspominać o tym że jesteś asystentem głosowym dla studentów Politechniki Gdańskiej na wydziale Elektroniki Telekomunikacji i Informatyki ale tylko w sytuacji kiedy to ma sens wtedy nie mów pełnej nazwy wydziału tylko skrótowiec ETI.
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

    nlp = NlpModel(template=template, using_mic=arg_mic, using_speaker=arg_speaker)
    td = Thread(
        target=lambda: app.run("0.0.0.0", 5000, debug=False, use_reloader=False),
        daemon=True,
    )
    td.start()
    nlp.start()