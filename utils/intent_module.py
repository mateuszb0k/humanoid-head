import spacy

class IntentDetector:
    def __init__(self):
        self.nlp = spacy.load("pl_core_news_sm")

        self.pg_keywords = {
            "sala", "plan", "zajęcie", "wykład", "ćwiczenie",
            "laboratorium", "uczelnia", "weti", "pg", "harmonogram",  "doktor", "inżynier", "profesor", "dojść",

        }

        self.weather_keywords = {
            "pogoda", "temperatura", "deszcz", "parasol","słonecznie", "deszczowo",
            "słońce", "wiatr", "prognoza", "zimno", "ciepło", "stopnie", "stopień"
        }

    def detect_intent(self, text: str) -> str:
        """
        This function takes in whole sentence, and returns the intent.
        """
        doc = self.nlp(text.lower()) # Split sentence into tokens

        lemmas = {token.lemma_ for token in doc if (not token.is_punct and not token.is_stop)} # Creating set for fast searching

        if lemmas & self.pg_keywords:
            return "PG"
        elif lemmas & self.weather_keywords:
            return "POGODA"
        else:
            return "OGOLNA"


if __name__ == "__main__":
    detector = IntentDetector()

    test_sentences = [
        "Jak doszła do sklepiku",
        "Gdzie znajdę doktora Czubenko?",
        "Co tam u Ciebie?",
        "Ile jest stopni?"
    ]

    for sentence in test_sentences:
        intent = detector.detect_intent(sentence)
        print(f"Sentence: {sentence} -> {intent}")
