# SPDX-FileCopyrightText: 2023-present Massimiliano Pippi <mpippi@gmail.com>
#
# SPDX-License-Identifier: MIT

CREATE_VECTORS_META = """
CREATE SEQUENCE vectors_meta_id START 1;
CREATE TABLE vectors_meta (
    id INTEGER PRIMARY KEY default nextval('vectors_meta_id'),
    vector_id INTEGER,
    text VARCHAR,
    embeddings DOUBLE[],
);
""".strip()

INSERT_VECTORS_META = """
INSERT INTO vectors_meta(vector_id, text, embeddings) VALUES ($vector_id, $text, $embeddings);
""".strip()

SELECT_VECTORS_META = """
SELECT (vector_id, text, embeddings) FROM vectors_meta WHERE vector_id == $vector_id
"""
