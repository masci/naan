# SPDX-FileCopyrightText: 2023-present Massimiliano Pippi <mpippi@gmail.com>
#
# SPDX-License-Identifier: MIT

CREATE_VECTORS_META = """
CREATE TABLE vectors (
    vector_id INTEGER PRIMARY KEY,
    text VARCHAR,
    embeddings DOUBLE[],
);
CREATE SEQUENCE vectors_meta_id START 1;
CREATE TABLE vectors_meta (
    id INTEGER PRIMARY KEY default nextval('vectors_meta_id'),
    vector_id INTEGER REFERENCES vectors(vector_id),
    key VARCHAR,
    value UNION(num DOUBLE, str VARCHAR)
);
""".strip()

INSERT_VECTORS = """
INSERT INTO vectors(vector_id, text, embeddings)
VALUES ($vector_id, $text, $embeddings);
""".strip()

INSERT_META = """
INSERT INTO vectors_meta (vector_id, key, value)
VALUES ($vector_id, $key, $value)
"""

SELECT_VECTORS = """
SELECT (vector_id, text, embeddings) FROM vectors WHERE vector_id == $vector_id
""".strip()

SELECT_VECTORS_NO_EMBEDDINGS = """
SELECT (vector_id, text) FROM vectors WHERE vector_id == $vector_id
""".strip()


def get_select_query(return_embeddings: bool, filter: str | None = None):
    q = SELECT_VECTORS if return_embeddings else SELECT_VECTORS_NO_EMBEDDINGS
    return q
