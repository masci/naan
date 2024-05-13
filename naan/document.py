from dataclasses import dataclass


@dataclass
class Document:
    vector_id: int
    content: str
    embeddings: list[float] | None
