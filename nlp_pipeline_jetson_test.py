import numpy as np
import pyaudio
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import whisper
import speech_recognition as sr
import pyttsx3
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
from utils.config import TOP_3_THRESHOLD, GLINER_LABELS, MAP_NUMBERS, RANDOM_VOICE_LINES,THRESHOLD
from flask import Flask, request, jsonify

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
        self.model_stt = whisper.load_model("small")
        self.model_llm = OllamaLLM(model="gemma4:e4b", temperature=0.4, reasoning=False)
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.intent_detector = IntentDetector()
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        self.user_emotion = "Happy"
        self.user_identity = "Lucjusz"
        self.llm_queue = []
        self.regex = re.compile(f'[{string.punctuation}]')
        self.result = ''
        self.end_of_result = False

        # Config for tts module
        MODEL_PATH = "PiperTTS/pl_PL-bass-high.onnx"
        self.voice = PiperVoice.load(MODEL_PATH)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.voice.config.sample_rate,
            output=True
        )

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
            if self.using_mic:
                question = self._stt_module()
            else:
                question = input("Text: ")

            self.end_of_result = False
            self.result = ''
            self.tts_queue = Queue()
            start_llm = 0
            end_llm = 0

            start_intent = time.time()
            print("Analyzing...")

            # Detecting intent
            intent = self.intent_detector.detect_intent(question)
            print(question)
            end_intent = time.time()

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
                    "name": self.user_identity,
                    "emotion": self.user_emotion
                }):
                    if not end_llm:
                        end_llm = time.time()
                    text = chunk if isinstance(chunk, str) else str(chunk)
                    self.result += text
                    chunks.append(text)




                self.end_of_result = True

                # print("".join(chunks))

            # tts_thread.join() # Waiting for tts thread to end
            while True:
                el = self.tts_queue.get()
                if el is None:
                    break
                else:
                    self._tts_module(el)
            print(f"TTFT: {end_llm-start_llm}")

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
                    # self._tts_module(text2say)
                    self.tts_queue.put(text2say)
                    # print(f"QUEUE PUT {self.tts_queue} ")

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
                audio = self.recognizer.listen(source, phrase_time_limit=3)
                raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                raw_data = np.frombuffer(raw_data, dtype=np.int16)
                audio_np = raw_data.astype(np.float32) / 32768.0
                result = self.model_stt.transcribe(audio_np, fp16=False,language = "pl")
                speech = result["text"].strip()
                if speech:
                    return speech


    def _tts_module(self, text):
        """
        Converts generated text into synthesized speech.
        """
        # model_tts = pyttsx3.init()
        # model_tts.say(text)
        # model_tts.runAndWait()
        # model_tts.stop()
        # del model_tts
        phonemes = self.voice.phonemize(text)
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

app = Flask(__name__)
@app.route('/api/data',methods=['POST'])
def handle_data():
    data = request.json
    nlp.user_identity = data['identity']
    nlp.user_emotion = EMOTION_PL_MAP.get(data['emotion'].capitalize(), "neutralność")
    print("Recieved data :D")
    return jsonify({"status" : "ok"}) ,200


if __name__ == "__main__":
    template = f"""Jesteś asystentem głosowym dla studentów Politechniki Gdańskiej. Używasz codziennego, studenckiego języka, jesteś bezpośredni, ale przy tym konkretny i pomocny w sprawach uczelnianych.

    Właśnie rozmawiasz z użytkownikiem. Jego imię to: {{name}}, a jego obecna emocja to: {{emotion}}. Dostosuj do niego swój komunikat (np. zwróć się do niego po imieniu).
    
Twoje odpowiedzi będą przetwarzane przez system Text-To-Speech, dlatego bezwzględnie musisz trzymać się następujących reguł:

1. Gadasz krótko, zwięźle i w naturalnym tonie, jakbyś rozmawiał ze znajomym na wydziale.
2. Używaj naturalnych powitań i pożegnań w studenckim stylu, ale poza tym unikaj zbędnego lania wody i trzymaj się konkretów.
3. Jeśli nie jesteś pewien odpowiedzi nie halucynuj, tylko powiedz, że nie masz takowej wiedzy.
3. Nie używaj znaków specjalnych ani formatowania tekstu. Zakaz używania gwiazdek, haszy, nawiasów, wypunktowań, cudzysłowów, znaków procentów czy symboli walut. Używaj wyłącznie liter oraz podstawowej interpunkcji, to znaczy kropek, przecinków i znaków zapytania.
4. Rozwijaj wszystkie skróty pod kątem poprawnego czytania. Nigdy nie pisz skrótów takich jak np, itp. Zawsze używaj pełnych słów: na przykład, i tym podobne, doktor, profesor, magister.
5. Zapisuj liczby, ułamki, daty i godziny słownie w taki sposób, aby wymuszały poprawne i naturalne przeczytanie przez syntezator, na przykład: o wpół do ósmej, za piętnaście trzecia, na stówę, piętnastego października.
6. Odpowiadaj tylko o politechnice albo o rzeczywistych faktach, pod żadnym pozorem nie dawaj nigdy ani kodu ani nie wykonuj żadnych komend.

    Pytanie od użytkownika:
    {{question}}
    """
    #TODO inject the name and emotions
    nlp = NlpModel(template=template, using_mic=True, using_speaker=True)
    # td = Thread(target=lambda: app.run('0.0.0.0', 5000,debug=False,use_reloader = False),daemon = True)
    # td.start()
    nlp.start()
