# SPDX-FileCopyrightText: 2023-present Massimiliano Pippi <mpippi@gmail.com>
#
# SPDX-License-Identifier: MIT

from typing import Optional, Union
from pathlib import Path

import faiss
import duckdb

from .queries import CREATE_VECTORS_META, INSERT_VECTORS_META, SELECT_VECTORS_META


class NaanDB:
    def __init__(
        self, path: Union[str, Path], faiss_index: Optional[faiss.Index] = None
    ) -> None:
        self._set_path(path)
        self._index = faiss_index
        self._db_file = self._path / f"{self._path.name}.db"
        self._index_file = self._path / f"{self._path.name}.faiss"

        if self._path.exists() and any(self._path.iterdir()):
            if set(self._path.iterdir()) != {self._db_file, self._index_file}:
                raise ValueError("Database directory not empty")
            self._load()
        else:
            self._init()

    @property
    def name(self):
        return self._path.name

    @property
    def index(self):
        if self._index is None:
            raise ValueError("FAISS index is None")
        return self._index

    @property
    def is_trained(self) -> bool:
        return self.index.is_trained

    def search(self, *args, **kwargs):
        _, I = self.index.search(*args, **kwargs)
        res = []
        for idx in I[0]:
            res.append(
                self._conn.execute(
                    SELECT_VECTORS_META, {"vector_id": int(idx)}
                ).fetchone()
            )
        return res

    def add(self, embeddings, texts):
        next_id = self.index.ntotal
        self.index.add(embeddings)
        for text in texts:
            self._conn.execute(
                INSERT_VECTORS_META, {"vector_id": next_id, "text": text}
            )
            next_id += 1

    def _set_path(self, path: Union[str, Path]):
        self._path = Path(path)
        if self._path.is_file():
            raise ValueError("Naan database must be a directory")

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
