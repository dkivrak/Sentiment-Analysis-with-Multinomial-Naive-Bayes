from typing import List, Dict, Tuple
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class Preprocessor:
    def __init__(self, tokenizer, embedder, **kwargs):
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.label_encoder = LabelEncoder(labels={"negative": 0, "positive": 1})
        self.__dict__.update(kwargs)

    def prepare_data(self, data: Dict[str, str]) -> Tuple[List[float], List[int]]:
        X = [self.tokenizer.tokenize(text) for text in data["text"]]
        X = self.embedder.embed(X)

        y = [self.label_encoder.label2id[label] for label in data["label"]]

        return X, y


class LabelEncoder:
    def __init__(self, labels, **kwargs):
        self.id2label = {v: k for k, v in labels.items()}
        self.label2id = {k: v for k, v in labels.items()}
        self.__dict__.update(kwargs)


class PreprocessorObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        if "pretrained_path" in self.__dict__:
            self.__dict__.update(self.load(self.pretrained_path).__dict__)
        else:
            print(f"!! {type(self).__name__} is not pretrained. You may need to train it first.")

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


class Tokenizer(PreprocessorObject):
    def __init__(self, remove_stopwords: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.remove_stopwords = remove_stopwords

    def train(self, texts: List[str]):
        pass

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()

        text = re.sub(r"[^\w\s]", "", text)

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

        return tokens


class Embedder(PreprocessorObject):
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        super().__init__(**kwargs)

    def train(self, tokens_list: List[List[str]]):
        texts = [" ".join(tokens) for tokens in tokens_list]
        self.vectorizer.fit(texts)

    def embed(self, tokens_list: List[List[str]]) -> List[List[float]]:
        if self.vectorizer is None:
            raise ValueError("Embedder must be trained before embedding.")

        texts = [" ".join(tokens) for tokens in tokens_list]
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray().tolist()
