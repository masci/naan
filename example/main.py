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
    sentence_embeddings = model.encode(sentences)
    dim = sentence_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    db = NaanDB("naan.db", index, force_recreate=True)
    db.add(sentence_embeddings, sentences)


def query(q):
    model = SentenceTransformer("bert-base-nli-mean-tokens")

    db = NaanDB("naan.db")

    k = 4
    query_embeddings = model.encode([q])
    return db.search(query_embeddings, k)


if __name__ == "__main__":
    index_data()
    for res in query("Someone sprints with a football"):
        print(res)
    # (5799, 'Two girls are laughing and other girls are watching them')
    # (20303, 'A group of football players is running in the field')
    # (14418, 'Four boys are sitting in a muddy stream.')
    # (28922, 'A group of people playing football is running in the field')
