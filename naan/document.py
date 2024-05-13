from dataclasses import dataclass, field


@dataclass
class Document:
    vector_id: int
    content: str
    embeddings: list[float] | None = field(default=None)
