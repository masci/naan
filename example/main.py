import json


from sentence_transformers import SentenceTransformer
import faiss

from naan import NaanDB


def load_data():
    sentences = []
    with open("sentences.json") as f:
        sentences = json.load(f)
    return sentences


def index_data():
    sentences = load_data()
    # initialize sentence transformer model
    model = SentenceTransformer("bert-base-nli-mean-tokens")
    # create sentence embeddings
    sentence_embeddings = model.encode(sentences[:100])
    print(sentence_embeddings.shape)
    dim = sentence_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    db = NaanDB("foo", index, force_recreate=True)
    db.add(sentence_embeddings, sentences)


def query(q):
    model = SentenceTransformer("bert-base-nli-mean-tokens")

    db = NaanDB("foo")
    k = 2
    query_embeddings = model.encode([q])
    print(query_embeddings.shape)

    # # D, I = index.search(xq, k)  # search
    return db.search(query_embeddings, k)  # search


if __name__ == "__main__":
    # index_data()
    print(query("The cat is black"))
