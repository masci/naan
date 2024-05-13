import pytest

from naan.filesystem import StorageFolder


def test_defaults(tmp_path):
    storage = StorageFolder(tmp_path / "test")
    assert storage.db_file.name == "test.db"
    assert storage.index_file.name == "test.faiss"
    assert storage.name == "test"


def test_init_path_is_file(tmp_path):
    with open(tmp_path / "test", "w") as f:
        f.write("test")

    with pytest.raises(FileExistsError):
        StorageFolder(tmp_path / "test")


def test_init_folder_exists_empty(tmp_path):
    p = tmp_path / "test"
    p.mkdir()
    storage = StorageFolder(tmp_path / "test")
    assert storage.db_file.name == "test.db"
    assert storage.index_file.name == "test.faiss"


def test_init_folder_exists_not_empty(tmp_path):
    p = tmp_path / "test"
    p.mkdir()
    with open(tmp_path / "test" / "foo", "w") as f:
        f.write("test")

    with pytest.raises(ValueError):
        StorageFolder(tmp_path / "test")

    # force overwrite
    StorageFolder(tmp_path / "test", force=True)


def test_ready(tmp_path):
    storage = StorageFolder(tmp_path / "test")
    assert not storage.ready
    with open(tmp_path / "test" / "test.faiss", "w") as f:
        f.write("test")
    assert not storage.ready
    with open(tmp_path / "test" / "test.db", "w") as f:
        f.write("test")
    assert storage.ready
