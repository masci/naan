# SPDX-FileCopyrightText: 2023-present Massimiliano Pippi <mpippi@gmail.com>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import duckdb
import faiss

from .queries import CREATE_VECTORS_META, INSERT_VECTORS_META, SELECT_VECTORS_META


class NaanDB:
    def __init__(
        self, path: str | Path, faiss_index: faiss.Index | None = None
    ) -> None:
        self._set_path(path)
        self._index = faiss_index
        self._db_file = self._path / f"{self._path.name}.db"
        self._index_file = self._path / f"{self._path.name}.faiss"

        if self._path.exists() and any(self._path.iterdir()):
            if set(self._path.iterdir()) != {self._db_file, self._index_file}:
                msg = "Database directory not empty"
                raise ValueError(msg)
            self._load()
        else:
            self._init()

    @property
    def name(self):
        return self._path.name

    @property
    def index(self):
        if self._index is None:
            msg = "FAISS index is None"
            raise ValueError(msg)
        return self._index

    @property
    def is_trained(self) -> bool:
        return self.index.is_trained

    def search(self, *args, **kwargs):
        _, indices = self.index.search(*args, **kwargs)
        return [
            self._conn.execute(SELECT_VECTORS_META, {"vector_id": int(idx)}).fetchone()
            for idx in indices[0]
        ]

    def add(self, embeddings, texts):
        next_id = self.index.ntotal
        self.index.add(embeddings)
        for text in texts:
            self._conn.execute(
                INSERT_VECTORS_META, {"vector_id": next_id, "text": text}
            )
            next_id += 1

    def _set_path(self, path: str | Path):
        self._path = Path(path)
        if self._path.is_file():
            msg = "Naan database must be a directory"
            raise ValueError(msg)

    def _init(self):
        self._path.mkdir()
        self._conn = duckdb.connect(database=str(self._db_file))
        self._conn.execute(CREATE_VECTORS_META)
        faiss.write_index(self.index, str(self._index_file))

    def _load(self):
        self._conn = duckdb.connect(database=str(self._db_file))
        self._index = faiss.read_index(str(self._index_file))

    def _insert_vector(self):
        pass
