import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import whisper
import speech_recognition as sr
import pyttsx3
from weather import get_weather,weather_prompt
import json

class NlpModel:
    """
    This class manages the voice assistant model. It integrates speech recognition (Whisper),
    LLM (gemma3:4b) and speech synthesis (pyttsx3)
    """
    def __init__(self, template = None):
        # The models may change in the future
        self.model_stt = whisper.load_model("small")
        self.model_llm = OllamaLLM(model="gemma3:4b")
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        # Setting prompt for LLM
        if template is not None:
            self.prompt = ChatPromptTemplate.from_template(template)
        else:
            template = "Tutaj jest pytanie do Ciebie: {question}"
            self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.model_llm
        self.tts = pyttsx3.init()

    def start(self):
        """
        STT -> LLM -> TTS
        The process runs indefinetely unless it is interrupted.
        """
        while True:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source)
                # The loop continues until the sound is recorded
                # STT phase
                while True:
                    print("Listening...")
                    audio = self.recognizer.listen(source, phrase_time_limit=3)
                    raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    raw_data = np.frombuffer(raw_data, dtype = np.int16)
                    audio_np = raw_data.astype(np.float32) / 32768.0
                    result = self.model_stt.transcribe(audio_np, fp16 = False)
                    question = result["text"].strip()
                    if question:
                        print("Analyzing...")
                        break

            # LLM phase
            result = self.chain.invoke({"question": question})
            # TTS phase
            self.tts.say(result)
            self.tts.runAndWait()
            self.tts.stop()


if __name__ == "__main__":
    weather_text_prompt = weather_prompt()  # defaults to gdansk
    rooms_data = {
        "EA": {
            "107": {"floor": "1",
                    "directions": "pojedź windą lub pójdź schodami na {floor} piętro i pójdź holem w lewo"},
            "240": {"floor": "2",
                    "directions": "pojedź windą lub pójdź schodami na {floor} piętro i pójdź holem w prawo"}
        },
        "NE": {
            "105": {"floor": "1",
                    "directions": "Pójdź w lewo od głównego wejścia, po schodach lub windą udaj się na {floor} piętro i wejdź w korytarz po lewej stronie"},
            "215": {"floor": "2",
                    "directions": "Pójdź na prawo od głównego wejścia, po schodach lub windą udaj się na {floor} piętro i wejdź w korytarz po prawej stronie"}
        }
    }

    rooms_context = json.dumps(rooms_data, ensure_ascii=False, indent=2)

    # Zabezpieczenie nawiasów klamrowych przed parserem LangChaina
    rooms_context_escaped = rooms_context.replace("{", "{{").replace("}", "}}")
    weather_text_escaped = weather_text_prompt.replace("{", "{{").replace("}", "}}")

    template = f"""Jesteś wszechstronnym asystentem studentów Politechniki Gdańskiej. 

    Instrukcje zachowania:
    1. Pytania o plan zajęć, uczelnię i lokalizację sal opieraj wyłącznie na poniższej bazie wiedzy.
    2. Gdy podajesz wskazówki dojścia do sali, obowiązkowo podmień tekst '{{{{floor}}}}' na odpowiedni numer piętra z danych.
    3. Pytania o pogodę opieraj na następujących danych: {weather_text_escaped}
    4. Pytania niezwiązane z uczelnią (np. programowanie, ogólna wiedza) traktuj jak standardowy sztuczna inteligencja, korzystając z własnej wiedzy.
    5. Odpowiadaj naturalnie i unikaj zbędnych powitań.

    Baza wiedzy o salach (Format: Budynek -> Numer sali -> Szczegóły):
    {rooms_context_escaped}

    Pytanie od użytkownika:
    {{question}}
    """

    nlp = NlpModel(template=template)
    nlp.start()