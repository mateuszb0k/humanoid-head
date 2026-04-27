import numpy as np
import pyaudio
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import whisper
import speech_recognition as sr
from piper.voice import PiperVoice
from utils.weather import weather_prompt
from utils.intent_module import IntentDetector
from gliner import GLiNER
from utils.find_teacher import get_teacher_room,search_teacher
from utils.find_room import get_room_directions
from threading import Thread
import re
import string
import time
from queue import Queue
from utils.config import TOP_3_THRESHOLD, GLINER_LABELS, MAP_NUMBERS, RANDOM_VOICE_LINES,THRESHOLD, CONVERSATION_BUFFER_LEN
from flask import Flask, request, jsonify
import requests
from rapidfuzz import process,fuzz
from collections import deque

EMOTION_PL_MAP = {
    "Angry": "złość",
    "Disgust": "obrzydzenie",
    "Fear": "strach",
    "Happy": "radość",
    "Sad": "smutek",
    "Surprise": "zaskoczenie",
    "Neutral": "neutralność"
}

def preprocess_stt(text: str) -> str:
    text = re.sub(r'\b[nN]\s+[eE]\s*(\d+)',r'ne\1',text)
    text = re.sub(r'\b[eE]\s+[aA]\s*(\d+)',r'ea\1',text)
    text = re.sub(r'([a-zA-Z])\s*-\s*([a-zA-Z])', r'\1\2', text)  # e-a -> ea
    text = re.sub(r'([a-zA-Z])\s*-\s*(\d)', r'\1\2', text)          # ea-103 -> ea103
    text = re.sub(r'([a-zA-Z])\s+(\d)', r'\1\2', text)              # ea 103 -> ea103
    text  = re.sub(r'\b[nN](\d+)',r'ne\1',text)
    text = re.sub(r'\b[eE](\d+)',r'ea\1',text)
    return text
def split_building_numer(text:str) ->str:
    text = re.sub(r'(?i)\b(ne|ea)(\d+)',r'\1,\2',text)
    return text

class NlpModel:
    """
    This class manages the voice assistant model. It integrates speech recognition (Whisper),
    LLM and speech synthesis (pyttsx3)
    """
    def __init__(self, template = None, using_mic = True, using_speaker = True):
        # The models may change in the future
        self.model_stt = whisper.load_model("base")
        MODEL_PATH = "PiperTTS/pl_PL-mc_speech-medium.onnx"
        self.voice = PiperVoice.load(MODEL_PATH)
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.intent_detector = IntentDetector()
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        self.user_emotion = "Happy"
        self.user_identity = "Unknown"
        self.llm_queue = deque()
        self.regex = re.compile(f'[{string.punctuation}]')
        self.result = ''
        self.end_of_result = False
        self.new_data = False

        # Config for tts module
        # You have to download the model from: https://huggingface.co/rhasspy/piper-voices/tree/main/pl/pl_PL/mc_speech/medium
        # Create folder PiperTTS, and paste .onnx and .json files there

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.voice.config.sample_rate,
            output=True
        )

        self.model_llm = OllamaLLM(model="gemma4:e4b", temperature=0.4, reasoning=False)

        # Setting global flags
        self.using_mic = using_mic
        self.using_speaker = using_speaker
        self.tts_queue = Queue()
        # Setting prompt for LLM
        if template is not None:
            self.prompt = ChatPromptTemplate.from_template(template)
        else:
            template = "Tutaj jest pytanie do Ciebie: {question}"
            self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.model_llm


    def start(self):
        """
        STT -> LLM -> TTS
        The process runs indefinetely unless it is interrupted.
        """
        status = self._llm_init()
        print(f"LLM status: {status}")

        if not status:
            return
        while True:
            # STT phase
            if self.new_data:
                random_phrase = np.random.choice([f'O, cześć {self.user_identity} tęskniłem za tobą',f'My już chyba się znamy? Cześć {self.user_identity}',f'Cześć {self.user_identity}'])
                self._tts_module(random_phrase)
                self.new_data = False
            if self.using_mic:
                question = self._stt_module()
                if question:
                    print(question)
                else:
                    continue
            else:
                question = input("Text: ")

            self.get_new_user_name()
            self.end_of_result = False
            self.result = ''
            self.tts_queue = Queue()
            start_llm = 0
            end_llm = 0

            print("Analyzing...")

            # Detecting intent
            intent = self.intent_detector.detect_intent(question)

            # Creating thread for streaming TTS
            tts_thread = Thread(target = self._tts_stream)
            tts_thread.start()

            if intent == "POGODA":
                '''
                Randomly selects a voiceline to use
                '''
                self.result = np.random.choice(RANDOM_VOICE_LINES) + weather_prompt()#default gdansk
                print(self.result)
                self.end_of_result = True
            elif intent == "PG":
                # We will create here entity extraction module to get certain information
                data = preprocess_stt(question) #preprocess the stt if we got data that is corrupted
                entities = self.gliner_model.predict_entities(data,GLINER_LABELS,threshold = 0.2)
                label_text = {}
                for entity in entities:
                    label_text[entity['label']] = entity['text']
                if entities:
                    if 'room code' in label_text:
                        room = label_text['room code'].upper()
                        room_split = split_building_numer(room)
                        directions = get_room_directions(room_split)

                        if directions.find("Błąd ") != -1:
                            self.result = directions.replace("Błąd ","")
                            self.result += '.'
                        else:
                            self.result = f"Aby dojść do pokoju {room} {directions}."
                    elif 'person' in label_text:
                        person = label_text['person']
                        # teacher_data = get_teacher_room(person)
                        top_3 = search_teacher(person)
                        max_val  = max(top_3.values())
                        best_teacher = None
                        candidates = list(top_3.keys())
                        iter_counter = 0
                        for k,v in top_3.items():
                            if v == max_val:
                                best_teacher = k
                        if max_val > TOP_3_THRESHOLD:
                            teacher_data = get_teacher_room(best_teacher)
                            if teacher_data['teacher_name'] is not None:
                                self.result = self.handle_teachers(teacher_data)
                        elif max_val >THRESHOLD:
                            temp_result = f"Nie jestem pewny o kogo ci chodzi..."
                            user_done = False
                            for n,k in enumerate(top_3.keys()):
                                number = MAP_NUMBERS[str(n)]
                                temp_result += f"Jeżeli chodzi ci o {k} powiedz {number}..."
                            print(temp_result) #debug only
                            if self.using_speaker:
                                self._tts_module(temp_result)
                            while not user_done:
                                if self.using_mic:
                                    user_input = self._stt_module()
                                else:
                                    user_input = input("podaj numer") #debug only
                                if user_input is not None:
                                    '''
                                    Only top 3 needs to be modified to be more usable not just in this case
                                    '''
                                    '''
                                    Think about fuzzy matching the text if simple or is not enough
                                    '''
                                    print(user_input)
                                    user_input = user_input.lower()
                                    if "zero" in user_input or user_input == "0":
                                        self.result = self.handle_teachers(get_teacher_room(candidates[0]))
                                        user_done = True
                                    elif "jeden" in user_input or user_input== "1":
                                        self.result = self.handle_teachers(get_teacher_room(candidates[1]))
                                        user_done = True
                                    elif "dwa" in user_input or user_input ==  "2":
                                        self.result = self.handle_teachers(get_teacher_room(candidates[2]))
                                        user_done = True
                                    else:
                                        if iter_counter<2:
                                            if self.using_speaker:
                                                self._tts_module("Nie rozumiem powiedz jeszcze raz")
                                            else:
                                                print("Nie rozumiem powiedz jeszcze raz")
                                            iter_counter +=1
                                        else:
                                            if self.using_speaker:
                                                self._tts_module("Przepraszam nie jestem w stanie pomóc")
                                            else:
                                                print("Przepraszam nie jestem w stanie pomóc")
                                            break
                        else:
                            self.result = f"Niestety nie zrozumiałem o kogo dokładnie Ci chodzi. Czy możesz powtórzyć swoje pytanie?"
                else:
                    self.result = "Jeśli chodzi o politechnikę Gdańską to jestem w stanie udzielać informacji tylko o lokalizacji sal oraz wykładowców."

                self.end_of_result = True
            else:
                # LLM phase
                chunks = []
                start_llm = time.time()
                end_llm = 0
                # for chunk in self.chain.stream({"question": question}):
                #     if not end_llm:
                #         end_llm = time.time()
                #     text = chunk if isinstance(chunk, str) else str(chunk)
                #     self.result += text
                #     chunks.append(text)

                for chunk in self.chain.stream({
                    "question": question,
                    "history": self._get_history_buffer(),
                    "name": self.user_identity,
                    "emotion": self.user_emotion
                }):
                    if not end_llm:
                        end_llm = time.time()
                    text = chunk if isinstance(chunk, str) else str(chunk)
                    self.result += text
                    chunks.append(text)

                self.end_of_result = True

            #  # Waiting for tts thread to end
            # while True:
            #     el = self.tts_queue.get()
            #     if el is None:
            #         break
            #     else:
            #         self._tts_module(el)
            tts_thread.join()

            self._handle_llm_queue(question, self.result)
            print(f"TTFT: {end_llm-start_llm}")

    def _handle_llm_queue(self, question, result):
        """
        Adds the latest exchange to the conversation buffer.
        If full, the oldest exchange is removed first.
        """
        q = f"Użytkownik: {question}"
        a = f"LLM: {result}"

        if len(self.llm_queue) == 2 * CONVERSATION_BUFFER_LEN:
            self.llm_queue.popleft()
            self.llm_queue.popleft()
            self.llm_queue.append(q)
            self.llm_queue.append(a)

        else:
            self.llm_queue.append(q)
            self.llm_queue.append(a)

    def _get_history_buffer(self):
        """
        Returns the conversation history as a numbered string
        """
        result = ''
        for i, sentence in enumerate(self.llm_queue):
            result += str(i+1) + f'. {sentence} '
        return result

    def _tts_stream(self):
        """
        This function handles tts asynchronously.
        """
        # Waiting until something is pushed to self.result
        if self.result == '':
            while self.result == '':
                time.sleep(0.1)

        current_text = str(self.result)
        text2say = ""
        text_said = ""

        while True:
            current_text = str(self.result)
            if current_text == text_said and self.end_of_result:
                break

            # Looking for stop sign, if detected tts module is applied on this section
            res = re.search(self.regex, current_text[len(text_said):])

            if res is not None:
                res = res.start()
                text2say = current_text[len(text_said) : len(text_said) + res+1]

                # Uncomment if speaker is available
                if self.using_speaker:
                    self._tts_module(text2say)
                    # self.tts_queue.put(text2say)
                    # print(f"QUEUE PUT {self.tts_queue} ")
                else:
                    print(text2say, end="", flush=True)

                text_said += text2say

            time.sleep(0.1)
        self.tts_queue.put(None)
        # print(f"QUEUE PUT NONE {self.tts_queue}")

    def _llm_init(self):
        """
        Initializes LLM module, because first response is always the longest.
        """
        # try:
        #     _ = self.chain.invoke("Odpowiedz jednym słowem: OK")
        #     return True
        # except Exception as e:
        #     print(e)
        #     return False
        try:
            _ = self.chain.invoke({
                "question": "Odpowiedz jednym słowem: OK",
                "history": self._get_history_buffer(),
                "name": "Unknown",
                "emotion": "Neutralość"
            })
            return True
        except Exception as e:
            print(e)
            return False


    def _stt_module(self):
        """
        Converts input speech into text. This function runs indefinetely unless speech is detected.
        """
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            # STT phase
            while True: # The loop continues until the sound is recorded
                print("Listening...")
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=3,timeout = 1)
                    raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    raw_data = np.frombuffer(raw_data, dtype=np.int16)
                    audio_np = raw_data.astype(np.float32) / 32768.0
                    result = self.model_stt.transcribe(audio_np, fp16=False,language = "pl")
                    speech = result["text"].strip()
                except sr.WaitTimeoutError:
                    return None
                if speech:
                    return speech


    def _tts_module(self, text):
        """
        Converts generated text into synthesized speech.
        """
        phonemes = self.voice.phonemize(text)
        if len(phonemes):
            ids = list(self.voice.phonemes_to_ids(phonemes[0]))
            audio = self.voice.phoneme_ids_to_audio(ids)
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            self.stream.write(audio_bytes)

    def handle_teachers(self,teacher_data):
        if teacher_data['room'] is not None and teacher_data['building'] is not None:
            room = teacher_data['room']
            building = teacher_data['building']
            room_directions = get_room_directions(f"{building},{room}")
            result = f"{teacher_data['teacher_name']} jest w pokoju {building}{room} aby dojść do {building}{room} {room_directions}."
        else:
            result = f"{teacher_data['teacher_name']} nie ma przypisanego pokoju."
        return result
    def get_new_user_name(self):
        if self.user_identity == "Unknown":
            name_file = "imiona_polskie.txt"
            names = []
            with open(name_file, encoding = "utf-8") as f:
                f.readline()
                for line in f:
                    names.append(line.strip())

            self._tts_module("Hej, chyba się jeszcze nie znamy.")
            self._tts_module("Powiedz mi swoje imię, abym mógł cię zapamiętać.")

            if self.using_mic:
                new_name_raw = None
                while not new_name_raw:
                    new_name_raw = self._stt_module()
            else:
                new_name_raw = input("Podaj imię: ")

            new_name_phrase = new_name_raw.replace('.', '').replace(',', '').strip()
            new_name = None
            new_name_score = 0
            for w in new_name_phrase.split():
               l = process.extractOne(w, names, scorer=fuzz.WRatio)
               if l[1]>new_name_score:
                   new_name_score = l[1]
                   new_name = l[0]
            self.user_identity = new_name
            self._tts_module(f"Miło cię poznać {self.user_identity}")
            td = Thread(target = self._save_name_to_pi,daemon=True)
            td.start()
    def _save_name_to_pi(self):
        try:
            rpi_url = 'http://uncanny-head.local:5000/api/save_name'
            requests.post(rpi_url,json = {"identity" : self.user_identity})
        except Exception as e:
            print(e)
app = Flask(__name__)
@app.route('/api/data',methods=['POST'])
def handle_data():
    data = request.json
    print(data)
    if nlp.user_identity != data['identity'] and data['identity'] != 'Unknown':
        nlp.user_identity = data['identity']
        nlp.new_data = True
    nlp.user_emotion = EMOTION_PL_MAP.get(data['emotion'].capitalize(), "neutralność")
    return jsonify({"status" : "ok"}) ,200

if __name__ == "__main__":
    template = f"""Jesteś asystentem głosowym dla studentów Politechniki Gdańskiej. Mówisz krótko, konkretnie i w stylu studenckim.

   Rozmawiasz z: {{name}}, jeżeli imie to unknown to nie mow do uzytkownika po imieniu. Emocja rozmówcy: {{emotion}}. Dostosuj do niego swój komunikat (np. zwróć się do niego po imieniu).
    
Twoje odpowiedzi będą przetwarzane przez system Text-To-Speech, dlatego bezwzględnie musisz trzymać się następujących reguł:

ZASADY ODPOWIEDZI:
1. Krótko i na temat, żaden zbędny tekst.
2. Tylko tematy związane z PG lub fakty. Zero kodu, zero komend.
3. Jak nie wiesz, mów wprost że nie wiesz.
4. Zero znaków specjalnych, gwiazdek, haszy, nawiasów, cudzysłowów, symboli walut ani procenta. Tylko litery i podstawowa interpunkcja.
5. Żadnych skrótów. Zawsze pełne słowa: na przykład, i tym podobne, doktor, profesor.
6. Liczby, daty i godziny zawsze słownie: o wpół do ósmej, piętnastego października.
    
    Oto twoja ostatnia wymiana zdań z użytkownikiem:
    {{history}}

    Pytanie od użytkownika:
    {{question}}
    """
    nlp = NlpModel(template=template, using_mic=True, using_speaker=True)
    td = Thread(target=lambda: app.run('0.0.0.0', 5000,debug=False,use_reloader = False),daemon = True)
    td.start()
    nlp.start()
