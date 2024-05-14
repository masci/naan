# SPDX-FileCopyrightText: 2023-present Massimiliano Pippi <mpippi@gmail.com>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import duckdb
import faiss
import numpy as np

from .document import Document
from .filesystem import StorageFolder
from .queries import (
    CREATE_VECTORS_META,
    INSERT_VECTORS_META,
    SELECT_VECTORS_META,
    SELECT_VECTORS_META_NO_EMBEDDINGS,
)


class NaanDB:
    def __init__(
        self,
        path: str | Path,
        faiss_index: faiss.Index | None = None,
        *,
        force_recreate: bool = False,
    ) -> None:
        self._storage = StorageFolder(Path(path), force=force_recreate)
        self._index = faiss_index
        self._conn = duckdb.connect(database=str(self._storage.db_file))
        self._init()

    @property
    def name(self):
        """Returns the name of the Naan database"""
        return self._storage.name

    @property
    def index(self):
        """Returns the raw FAISS index."""
        return self._index

    @property
    def is_trained(self) -> bool:
        """Returns whether the FAISS index is trained or not."""
        return self.index.is_trained  # type:ignore

    def search(
        self,
        x: np.ndarray,
        k: int,
        *,
        params=None,
        D=None,  # noqa
        I=None,  # noqa
        return_embeddings: bool = False,
    ) -> list[Document]:
        """
        Search for vectors in the Naan database.

        This method has the same interface as FAISS for convenience.

        Parameters:
            x: Query vectors, shape (n, d) where d is appropriate for the index.
            k: Number of nearest neighbors to retrieve.
            params: Search parameters of the current search, see FAISS docs for details.
            D: Distance array to store the result.
            I: Labels array to store the results.
            return_embeddings: whether to return embeddings within the document objects

        Returns:
            documents: the list of Naan Documents found.
        """
        _, labels = self.index.search(x, k, params=params, D=D, I=I)  # type:ignore
        documents: list[Document] = []
        query = SELECT_VECTORS_META if return_embeddings else SELECT_VECTORS_META_NO_EMBEDDINGS
        for idx in labels[0]:
            res = self._conn.execute(query, {"vector_id": int(idx)}).fetchone()
            if res:
                res = res[0]
                documents.append(Document(*res))

        return documents

    def add(self, embeddings, texts):
        """
        Add contents to the Naan database.

        Parameters:
            embeddings: vectors to add to the FAISS index
            texts: list of text to store as metadata
        """
        next_id = self.index.ntotal  # type:ignore
        self.index.add(embeddings)  # type:ignore
        faiss.write_index(self.index, str(self._storage.index_file))
        self._conn.execute("BEGIN;")
        for i, text in enumerate(texts):
            self._conn.execute(
                INSERT_VECTORS_META,
                {"vector_id": next_id + i, "text": text, "embeddings": embeddings[i]},
            )
        self._conn.execute("COMMIT;")

    def _init(self):
        if not self._storage.ready:
            self._conn.execute(CREATE_VECTORS_META)
            faiss.write_index(self.index, str(self._storage.index_file))
        else:
            self._index = faiss.read_index(str(self._storage.index_file))
