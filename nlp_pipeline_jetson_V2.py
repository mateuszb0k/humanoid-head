import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# import whisper  <-- Zakomentowane dla testów tekstowych (oszczędność RAM)
# import speech_recognition as sr
# import pyttsx3
from utils.weather import weather_prompt
import json
from utils.intent_module import IntentDetector
from gliner import GLiNER
import re
from utils.find_teacher import get_teacher_room
from utils.find_room import get_room_directions

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
    This class manages the voice assistant model. (TEXT TEST MODE)
    """
    def __init__(self, template = None):
        # STT i MIC wyłączone na czas testów terminalowych, by nie obciążać RAM-u!
        # self.model_stt = whisper.load_model("small")
        # self.recognizer = sr.Recognizer()
        # self.mic = sr.Microphone()
        
        self.model_llm = OllamaLLM(model="gemma3:4b-it-qat")
        self.intent_detector = IntentDetector()

        # Setting prompt for LLM
        if template is not None:
            self.prompt = ChatPromptTemplate.from_template(template)
        else:
            template = "Tutaj jest pytanie do Ciebie: {question}"
            self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.model_llm


    def start(self):
        """
        TEXT INPUT -> LLM -> TEXT OUTPUT
        """
        print("Ładowanie modelu GLiNER (Encje)...")
        gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        print("Gotowe! Możesz zacząć pisać.")
        
        while True:
            # Wprowadzanie tekstu w terminalu zamiast _stt_module()
            question = input("\nTwoje pytanie (lub 'q' aby wyjść): ")
            
            if question.lower() in ['q', 'quit', 'exit', 'wyjdź']:
                print("Zamykanie programu...")
                break
                
            print("Analizowanie intencji...")

            # Detecting intent
            intent = self.intent_detector.detect_intent(question)
            print(f"[Wykryta intencja]: {intent}")

            if intent == "POGODA":
                result = weather_prompt() #default gdansk
            
            elif intent == "PG":
                # We will create here entity extraction module to get certain information
                data = preprocess_stt(question) #preprocess the stt if we got data that is corrupted
                entities = gliner_model.predict_entities(data,GLINER_LABELS,threshold = 0.2) 
                
                label_text = {}
                for entity in entities:
                    label_text[entity['label']] = entity['text']
                
                if entities:
                    if 'room code' in label_text:
                        room = label_text['room code'].upper()
                        room_split = split_building_numer(room)
                        directions = get_room_directions(room_split)

                        if directions.find("Błąd ") != -1:
                            result = directions.replace("Błąd ","")
                        else:
                            result = f"Aby dojść do pokoju {room} {directions}"
                    
                    elif 'person' in label_text:
                        person = label_text['person']
                        teacher_data = get_teacher_room(person)
                        if teacher_data['teacher_name'] is not None:
                            if teacher_data['room'] is not None and teacher_data['building'] is not None:
                                room = teacher_data['room']
                                building = teacher_data['building']
                                room_directions = get_room_directions(f"{building},{room}")
                                
                                # NAPRAWIONY BŁĄD TRASY DLA WYKŁADOWCY
                                if room_directions.find("Błąd ") != -1:
                                    room_directions = "ale nie mam w bazie instrukcji, jak tam dojść."
                                else:
                                    room_directions = f"aby dojść do {building}{room} {room_directions}"
                                    
                                result = f"{teacher_data['teacher_name']} jest w pokoju {building}{room}, {room_directions}"
                            else:
                                result = f"{teacher_data['teacher_name']} nie ma przypisanego pokoju."
                        else:
                            result = "Niestety nie zrozumiałem o kogo dokładnie Ci chodzi. Czy możesz powtórzyć swoje pytanie?"
                    
                    # NAPRAWIONY BŁĄD BRAKU ZMIENNEJ (UnboundLocalError)
                    else:
                        result = "Wykryłem obiekt, ale nie jest to ani sala, ani nazwisko."
                else:
                    result = "Jeśli chodzi o Politechnikę Gdańską, jestem w stanie udzielać informacji tylko o lokalizacji sal oraz wykładowców."
            
            else:
                # LLM phase
                print("Generowanie odpowiedzi przez LLM (Ollama)...")
                result = self.chain.invoke({"question": question})

            # Wypisywanie wyniku w terminalu zamiast TTS
            print(f"\n[ASYSTENT]: {result}\n")
            print("-" * 50)


if __name__ == "__main__":
    template = f"""Jesteś wszechstronnym asystentem studentów Politechniki Gdańskiej. 

    Instrukcje zachowania:
    1. Odpowiadaj naturalnie i unikaj zbędnych powitań.

    Pytanie od użytkownika:
    {{question}}
    """

    nlp = NlpModel(template=template)
    nlp.start()