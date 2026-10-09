"""Tiny RAG layer: TF-IDF retrieval over the Coorg crop-disease knowledge base."""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "diseases.json"

CROP_WORDS = {
    "coffee": ["coffee", "arabica", "robusta", "kaffi"],
    "pepper": ["pepper", "piper", "menasu"],
    "cardamom": ["cardamom", "elaichi", "yalakki", "elakki"],
}


def detect_crop(text):
    """Return 'coffee' / 'pepper' / 'cardamom' if the farmer mentions it, else None."""
    t = (text or "").lower()
    for crop, words in CROP_WORDS.items():
        if any(w in t for w in words):
            return crop
    return None


class KnowledgeBase:
    def __init__(self, path=DATA_PATH):
        with open(path, encoding="utf-8") as f:
            self.diseases = json.load(f)
        self.by_id = {d["id"]: d for d in self.diseases}
        docs = [self._doc(d) for d in self.diseases]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(docs)

    @staticmethod
    def _doc(d):
        # keywords repeated so they weigh more than long prose
        return " ".join(
            [d["name"], d["crop"], " ".join(d["aliases"]), d["symptoms"], d["cause"]]
            + [" ".join(d["keywords"])] * 2
        )

    def search(self, query, crop=None, top_k=3):
        """Return [(disease_dict, score), ...] best first."""
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        results = [
            (d, float(sims[i]))
            for i, d in enumerate(self.diseases)
            if not crop or d["crop"] == crop
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get(self, disease_id):
        return self.by_id.get(disease_id)
