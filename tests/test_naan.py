import duckdb
import faiss
import numpy as np
import pytest

from naan import NaanDB
from naan.__about__ import __version__


@pytest.fixture
def index():
    return faiss.IndexFlatL2(384)


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


def test_add(tmp_path, index):
    vectors = np.array([np.random.rand(384) for i in range(10)])
    texts = ["foo"] * 10
    db = NaanDB(tmp_path / "test", index)
    db.add(vectors, texts)
    tot = db._conn.execute("SELECT COUNT(*) FROM vectors;").fetchone()  # noqa
    assert tot == (10,)


def test_add_error_not_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(NaanDB, "_init", lambda _: True)
    monkeypatch.setattr(NaanDB, "is_trained", False)
    db = NaanDB(tmp_path / "test", None)
    with pytest.raises(ValueError, match="The index needs to be trained"):
        db.add([], [])


def test_add_size_mismatch(tmp_path, index):
    db = NaanDB(tmp_path / "test", index)
    with pytest.raises(ValueError, match="The number of embeddings must match"):
        db.add([1], [])

    with pytest.raises(
        ValueError,
        match="The number of metadata objects must match the number of texts",
    ):
        db.add([1], ["foo"], [])


def test_search(tmp_path, index):
    vectors = np.array([np.random.rand(384) for i in range(10)])
    texts = ["foo"] * 10
    db = NaanDB(tmp_path / "test", index)
    db.add(vectors, texts)
    res = db.search(np.array([np.random.rand(384)]), 3)
    assert len(res) == 3
    for doc in res:
        assert doc.embeddings is None


def test_search_w_embeddings(tmp_path, index):
    vectors = np.array([np.random.rand(384) for i in range(10)])
    texts = ["foo"] * 10
    db = NaanDB(tmp_path / "test", index)
    db.add(vectors, texts)
    res = db.search(np.array([np.random.rand(384)]), 3, return_embeddings=True)
    assert len(res) == 3
    for doc in res:
        assert doc.embeddings is not None


def test_search_too_many_k(tmp_path, index):
    vectors = np.array([np.random.rand(384) for i in range(3)])
    texts = ["foo"] * 3
    db = NaanDB(tmp_path / "test", index)
    db.add(vectors, texts)
    res = db.search(np.array([np.random.rand(384)]), 10)
    assert len(res) == 3
