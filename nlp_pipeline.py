import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import whisper
import speech_recognition as sr
import pyttsx3

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
            model_tts = pyttsx3.init()
            model_tts.say(result)
            model_tts.runAndWait()
            model_tts.stop()
            del model_tts

if __name__ == "__main__":
    template = """
        Jesteś osobą pomagającą naukowo oraz wszechstronnym asystentem.
        Każdą swoją wypowiedź, niezależnie od tematu, kończ bezwzględnie słowami: Tak jest proszę pana
        
        Instrukcje zachowania:
        1. Jeśli użytkownik pyta o plan zajęć, uczelnię, sale lub przedmioty, oprzyj swoją odpowiedź wyłącznie na poniższym planie zajęć.
        2. Jeśli użytkownik zadaje pytanie niezwiązane z uczelnią i planem zajęć (np. pyta o ogólną wiedzę, programowanie, ciekawostki), odpowiedz mu jak standardowy, pomocny chatbot, korzystając ze swojej ogólnej wiedzy.
        
        Oto plan zajęć studenta:
        Poniedziałek:
        - 08:00 - 09:30: Analiza matematyczna, sala 101 (Wydział Gmach Główny)
        - 10:00 - 11:30: Algorytmy i struktury danych, sala 202
        - 12:00 - 13:30: Fizyka, laboratorium 3B
        
        Wtorek:
        - 09:00 - 10:30: Bazy danych, sala 105
        - 11:00 - 12:30: Język angielski, sala 410
        
        Środa:
        - 14:00 - 15:30: Sieci komputerowe, sala 304
        
        Czwartek i Piątek:
        - Dni wolne od zajęć.
        
        Tutaj jest pytanie do Ciebie: {question}
    """
    nlp = NlpModel(template = template)
    nlp.start()