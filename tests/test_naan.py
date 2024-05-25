import shutil

import duckdb
import faiss
import numpy as np
import pytest

from naan import NaanDB
from naan.__about__ import __version__


@pytest.fixture
def index():
    return faiss.IndexFlatL2(384)


@pytest.fixture
def vectors():
    return np.array([np.random.rand(384) for _ in range(10)])


@pytest.fixture
def db(tmp_path, index):
    yield NaanDB(tmp_path / "test", index)
    shutil.rmtree(tmp_path / "test")


def test_version():
    assert __version__


def test_naan(tmp_path, index):
    db = NaanDB(tmp_path / "test", index)
    assert db.name == "test"
    assert db.index == index
    assert db.is_trained is True


def test_init_index_exists(tmp_path, index):
    p = tmp_path / "test"
    p.mkdir()
    faiss.write_index(index, str(p / "test.faiss"))
    duckdb.connect(database=str(p / "test.db"))
    db = NaanDB(p, index)
    # The index should be re-loaded from disk, hence a different instance
    assert db.index != index
    shutil.rmtree(p)


def test_add(db, vectors):
    texts = ["foo"] * len(vectors)
    db.add(vectors, texts)
    tot = db._conn.execute("SELECT COUNT(*) FROM vectors;").fetchone()  # noqa
    assert tot == (10,)


def test_add_list_of_floats(db):
    texts = ["foo"] * 5
    vectors = [[0] * 384] * 5
    db.add(vectors, texts)
    tot = db._conn.execute("SELECT COUNT(*) FROM vectors;").fetchone()  # noqa
    assert tot == (5,)


def test_add_metadata(db, vectors):
    texts = ["foo"] * len(vectors)
    common_metadata = {"tag": "test_add_metadata"}
    db.add(vectors, texts, common_metadata)
    results = db._conn.execute("SELECT * FROM vectors_meta;").fetchall()
    assert len(results) == 10
    for res in results:
        assert res[3] == "test_add_metadata"


def test_add_metadata_list(db, vectors):
    texts = ["foo"] * len(vectors)
    metadata = [{"tag": f"tag_{i}"} for i in range(10)]
    db.add(vectors, texts, metadata)
    results = db._conn.execute("SELECT * FROM vectors_meta;").fetchall()
    assert len(results) == 10
    for i, res in enumerate(results):
        assert res[3] == f"tag_{i}"


def test_add_error_not_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(NaanDB, "_init", lambda _: True)
    monkeypatch.setattr(NaanDB, "is_trained", False)
    db = NaanDB(tmp_path / "test", None)
    with pytest.raises(ValueError, match="The index needs to be trained"):
        db.add([], [])
    shutil.rmtree(tmp_path / "test")


def test_add_size_mismatch(db, vectors):
    with pytest.raises(ValueError, match="The number of embeddings must match"):
        db.add(vectors, [])

    with pytest.raises(
        ValueError,
        match="The number of metadata objects must match the number of texts",
    ):
        db.add(vectors, ["foo"] * 10, [])


def test_search(db):
    vectors = np.array([np.random.rand(384) for i in range(10)])
    texts = ["foo"] * 10
    db.add(vectors, texts)
    res = db.search(np.array([np.random.rand(384)]), 3)
    assert len(res) == 3
    for doc in res:
        assert doc.embeddings is None


def test_search_w_embeddings(db):
    vectors = np.array([np.random.rand(384) for i in range(10)])
    texts = ["foo"] * 10
    db.add(vectors, texts)
    res = db.search(np.array([np.random.rand(384)]), 3, return_embeddings=True)
    assert len(res) == 3
    for doc in res:
        assert doc.embeddings is not None


def test_search_too_many_k(db):
    vectors = np.array([np.random.rand(384) for i in range(3)])
    texts = ["foo"] * 3
    db.add(vectors, texts)
    res = db.search(np.array([np.random.rand(384)]), 10)
    assert len(res) == 3
