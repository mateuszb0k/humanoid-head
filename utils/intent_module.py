import spacy

class IntentDetector:
    def __init__(self):
        self.nlp = spacy.load("pl_core_news_sm")

        self.pg_keywords = {
            "sala", "gabinet", "laboratorium", "budynek",
            "wydział", "dziekanat", "piętro", "parter", "wejście", "pokój",
            "weti", "eti", "pg", "doktor", "profesor", "inżynier",
            "magister", "wykładowca", "prowadzący", "dziekan",
            "dojść", "znaleźć", "szukać", "trafić", "zaprowadzić","pokierować", 
            "mapa", "plan","sala ea","sala ne", "sali"
        }

        self.weather_keywords = {
            "pogoda", "temperatura", "deszcz", "parasol", "słońce",
            "wiatr", "prognoza", "zimno", "ciepło", "stopień",
            "śnieg", "burza", "chmura", "mróz", "upał", "padać",
            "wiać", "aura", "warunek", "niebo", "zewnątrz",
            "ciepły", "zimny", "słoneczny", "pochmurny", "deszczowy"
        }

        self.aud_keywords = {
            "audytorium", "wykładowa", "wykładowej", "wykładową", "wykładowe", "aula", "wykład",
            "biblioteka", "szatnia", "jadalnia", "bar", "stołówka", "czytelnia", "auditium", "auditorium",
            "lewe", "prawe" 
        }

    def detect_intent(self, text: str) -> str:
        """
        This function takes in whole sentence, and returns the intent.
        """
        doc = self.nlp(text.lower()) # Split sentence into tokens

        lemmas = {token.lemma_ for token in doc if (not token.is_punct and not token.is_stop)} # Creating set for fast searching

        if lemmas & self.aud_keywords:
            return "AUD"
        elif lemmas & self.weather_keywords:
            return "POGODA"
        elif lemmas & self.pg_keywords:
            return "PG"
        else:
            return "OGOLNA"


if __name__ == "__main__":
    detector = IntentDetector()

    test_sentences = [
        "Gdzie znajde pogodnego doktora?",
        "Gdzie znajdę doktora Czubenko?",
        "Co tam u Ciebie?",
        "Ile jest stopni?",
        "Jak ciepło dziś będzie?"
    ]

    for sentence in test_sentences:
        intent = detector.detect_intent(sentence)
        print(f"Sentence: {sentence} -> {intent}")
