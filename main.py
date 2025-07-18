import numpy as np 
import pandas as pd
import os
from preprocessing import Tokenizer, Embedder, Preprocessor
from model import Model

df = pd.read_csv("data/train.csv")

tokenizer = Tokenizer(remove_stopwords=True)
embedder = Embedder()

texts = df["text"].tolist()
labels = df["label"].tolist()
tokens = [tokenizer.tokenize(text) for text in texts]

tokenizer.train(texts)
embedder.train(tokens)

preprocessor = Preprocessor(tokenizer=tokenizer, embedder=embedder)
X, y = preprocessor.prepare_data(df)

if len(y) > 0 and isinstance(y[0], (list, np.ndarray)):
    y = np.argmax(y, axis=1)

model = Model()
model.train(X, y)

os.makedirs("saved_objects", exist_ok=True)

tokenizer.save("saved_objects/tokenizer.pkl")
embedder.save("saved_objects/embedder.pkl")
model.save("saved_objects/model.pkl")

print("Model have been trained and saved successfully.")
