# SPDX-FileCopyrightText: 2023-present Massimiliano Pippi <mpippi@gmail.com>
#
# SPDX-License-Identifier: MIT

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DB_FILENAME_PATTERN = "{}.db"
INDEX_FILENAME_PATTERN = "{}.faiss"


class StorageFolder:
    def __init__(self, path: Path, *, force: bool = False) -> None:
        try:
            path.mkdir(exist_ok=True)
        except FileExistsError as e:
            logger.exception("Path %s should be a directory", path.name)
            raise e from None

        self._path = path
        self._db_file = self._path / DB_FILENAME_PATTERN.format(path.name)
        self._index_file = self._path / INDEX_FILENAME_PATTERN.format(path.name)

        # We can assume at this point the path exists and is a directory.
        # If the folder is not empty, see if it's a valid storage
        contents = list(self._path.iterdir())
        if contents and not self.ready:
            if force:
                shutil.rmtree(self._path)
                self._path.mkdir()
            else:
                msg = "Directory not empty and not a Naan database"
                raise ValueError(msg)

    @property
    def ready(self) -> bool:
        return set(self._path.iterdir()) == {self._db_file, self._index_file}

    @property
    def db_file(self):
        return self._db_file

    @property
    def index_file(self):
        return self._index_file
