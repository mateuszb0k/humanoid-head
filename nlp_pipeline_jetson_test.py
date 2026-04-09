import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import whisper
import speech_recognition as sr
import pyttsx3
from utils.weather import weather_prompt
import json
import time
from utils.intent_module import IntentDetector
from gliner import GLiNER
import re
from utils.find_teacher import get_teacher_room
from utils.find_room import get_room_directions
from threading import Thread
import re
import string
import time

GLINER_LABELS  = [
    "room code",
    "person"
]
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
    LLM (gemma3:4b) and speech synthesis (pyttsx3)
    """
    def __init__(self, template = None):
        # The models may change in the future
        self.model_stt = whisper.load_model("small")
        self.model_llm = OllamaLLM(model="gemma4:e4b", temperature=0.1, reasoning=False)
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.intent_detector = IntentDetector()
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")


        self.llm_queue = []
        self.regex = re.compile(f'[{string.punctuation}]')
        self.result = ''
        self.end_of_result = False

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
            # question = self._stt_module()
            self.end_of_result = False
            self.result = ''
            start_llm = 0
            end_llm = 0
            question = input("Text: ")
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
                self.result = "Poczekaj sprawdzam pogodę." + "Szukam termometru." + "Własnie dokonuje pomiaru." + weather_prompt() #default gdansk
                self.end_of_result = True
            elif intent == "PG":
                # We will create here entity extraction module to get certain information
                data = preprocess_stt(question) #preprocess the stt if we got data that is corrupted
                entities = self.gliner_model.predict_entities(data,GLINER_LABELS,threshold = 0.2) #TODO fix gliner, because it is too slow
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
                        teacher_data = get_teacher_room(person)
                        if teacher_data['teacher_name'] is not None:
                            if teacher_data['room'] is not None and teacher_data['building'] is not None:
                                room = teacher_data['room']
                                building = teacher_data['building']
                                room_directions = get_room_directions(f"{building},{room}")
                                self.result = f"{teacher_data['teacher_name']} jest w pokoju {building}{room} aby dojść do {building}{room} {room_directions}."
                            else:
                                self.result = f"{teacher_data['teacher_name']} nie ma przypisanego pokoju."
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
                for chunk in self.chain.stream({"question": question}):
                    if not end_llm:
                        end_llm = time.time()
                    text = chunk if isinstance(chunk, str) else str(chunk)
                    self.result += text
                    chunks.append(text)

                self.end_of_result = True

                # print("".join(chunks))

            tts_thread.join() # Waiting for tts thread to end
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

                # Uncomment if microphone is available
                # self._tts_module(text2say)

                print(text2say, end="", flush=True)

                text_said += text2say

            time.sleep(0.1)

    def _llm_init(self):
        """
        Initializes LLM module, because first response is always the longest.
        """
        try:
            _ = self.chain.invoke("Odpowiedz jednym słowem: OK")
            return True
        except Exception as e:
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
                result = self.model_stt.transcribe(audio_np, fp16=False)
                speech = result["text"].strip()
                if speech:
                    return speech


    def _tts_module(self, text):
        """
        Converts generated text into synthesized speech.
        """
        model_tts = pyttsx3.init()
        model_tts.say(text)
        model_tts.runAndWait()
        model_tts.stop()
        del model_tts



if __name__ == "__main__":
    template = f"""Jesteś asystentem głosowym dla studentów Politechniki Gdańskiej.
    Twoje odpowiedzi czyta syntezator mowy (TTS). Przestrzegaj tych zasad:
    
    1. Odpowiadaj zwięźle, ale rozwiń myśl, jeśli sytuacja tego wymaga.
    2. Przechodź od razu do rzeczy. Całkowicie pomijaj powitania i pożegnania.
    3. Używaj wyłącznie liter i podstawowej interpunkcji. Żadnych znaków specjalnych, symboli czy nawiasów.
    4. Nigdy nie używaj skrótów. Pisz pełnymi słowami: na przykład, magister, Politechnika Gdańska.
    5. Liczby i daty zapisuj tak, aby wymuszały poprawne przeczytanie (np. wpół do ósmej, pierwszego października).

    Pytanie od użytkownika:
    {{question}}
    """

    nlp = NlpModel(template=template)
    nlp.start()