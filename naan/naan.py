# SPDX-FileCopyrightText: 2023-present Massimiliano Pippi <mpippi@gmail.com>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import duckdb
import faiss

from .queries import CREATE_VECTORS_META, INSERT_VECTORS_META, SELECT_VECTORS_META
from .filesystem import StorageFolder


class NaanDB:
    def __init__(
        self,
        path: str | Path,
        faiss_index: faiss.Index | None = None,
        force_recreate: bool = False,
    ) -> None:
        self._storage = StorageFolder(Path(path), force=force_recreate)
        self._index = faiss_index
        self._conn = duckdb.connect(database=str(self._storage.db_file))
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
        next_id = self.index.ntotal + 1
        self.index.add(embeddings)  # type:ignore
        faiss.write_index(self.index, str(self._storage.index_file))
        for text in texts:
            print(next_id)
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
        if not self._storage.ready:
            self._conn.execute(CREATE_VECTORS_META)
            faiss.write_index(self.index, str(self._storage.index_file))
        else:
            self._index = faiss.read_index(str(self._storage.index_file))
