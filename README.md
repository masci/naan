# naan

[![PyPI - Version](https://img.shields.io/pypi/v/naan.svg)](https://pypi.org/project/naan)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/naan.svg)](https://pypi.org/project/naan)

-----

**Table of Contents**

- [naan](#naan)
  - [What is Naan?](#what-is-naan)
  - [Installation](#installation)
  - [Index data](#index-data)
  - [License](#license)

## What is Naan?

- Naan is a wrapper around FAISS indexes that provides metadata storage and retrieval for the vectors added to the index.
- Naan's job is to eliminate the tedious task of keeping around the original content before it's encoded
and added to the index.
- Naan is NOT a vector database. All the vector-search operations are demanded to FAISS.

## Installation

```console
pip install naan
```

## Index data

To see Naan in action, let's first get some data to embed:

```python
from io import StringIO
import requests
import json


res = requests.get("https://raw.githubusercontent.com/masci/naan/main/example/sentences.json")
sentences = json.load(StringIO(res.text))
```

Naan tries not to get in the way you manage your FAISS index, so the first step is always setting
up the FAISS side of things:

```python
from sentence_transformers import SentenceTransformer
import faiss


model = SentenceTransformer("bert-base-nli-mean-tokens")
sentence_embeddings = model.encode(sentences)
dim = sentence_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
```

Now it's time to wrap the FAISS index with Naan and use it to index data:

```python
from naan import NaanDB


# Create a Naan database from scratch
db = NaanDB("db.naan", index, force_recreate=True)
db.add(sentence_embeddings, sentences)
```

Naan will add the vector embeddings to the FAISS index, and will also store the original sentences.
This way, a vector search will look like this:

```python
# Reopen an existing Naan database
db = NaanDB("db.naan")
query_embeddings = model.encode(["The book is on the table"])
# Naan's search API is the same as FAISS, let's get the 3 closest vectors
results = db.search(query_embeddings, 3)
for result in results:
    print(result)
# Document(vector_id=11451, content='A group of people sitting around a desk.', embeddings=None)
# Document(vector_id=2754, content='A close-up picture of a desk with a computer and papers on it.', embeddings=None)
# Document(vector_id=11853, content='A computer on a desk.', embeddings=None)
```

## License

`naan` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
