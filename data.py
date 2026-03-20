import pandas as pd
import numpy as np

Vocabulary = {"No": 0, "i": 1, "am": 2, "your": 3, "father": 4, ".": 5}

tokens = ["No", "i", "am", "your", "father", "."]

ids = [Vocabulary[token] for token in tokens]

dfvocab = pd.DataFrame(list(Vocabulary.items()), columns=["word", "id"])

sizevocab = len(Vocabulary)

d_model = 64

embedding_matrix = np.random.rand(sizevocab, d_model)

sentenceembeddings = [embedding_matrix[id] for id in ids]

X = np.array(sentenceembeddings)
X = X.reshape(1, len(tokens), d_model)

if __name__ == "__main__":

    print(dfvocab)
    print("ids:", ids)
    print("embedding_matrix shape:", embedding_matrix.shape)
    print(sentenceembeddings)
    print("X shape:", X.shape)