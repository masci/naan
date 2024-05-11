CREATE_VECTORS_META = """
CREATE TABLE vectors_meta (id INTEGER PRIMARY KEY, text VARCHAR);
""".strip()

INSERT_VECTORS_META = """
INSERT INTO vectors_meta VALUES ($vector_id, $text);
""".strip()

SELECT_VECTORS_META = """
SELECT id, text FROM vectors_meta WHERE id == $vector_id
"""
