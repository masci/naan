import pytest
import faiss
import duckdb
import numpy as np

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
    tot = db._conn.execute("SELECT COUNT(*) FROM vectors_meta;").fetchone()
    assert tot == (10,)


def test_search(tmp_path, index):
    vectors = np.array([np.random.rand(384) for i in range(10)])
    texts = ["foo"] * 10
    db = NaanDB(tmp_path / "test", index)
    db.add(vectors, texts)
    res = db.search(np.array([np.random.rand(384)]), 3)
    assert len(res) == 3


def test_search_too_many_k(tmp_path, index):
    vectors = np.array([np.random.rand(384) for i in range(3)])
    texts = ["foo"] * 3
    db = NaanDB(tmp_path / "test", index)
    db.add(vectors, texts)
    res = db.search(np.array([np.random.rand(384)]), 10)
    assert len(res) == 3
