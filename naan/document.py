from dataclasses import dataclass


@dataclass
class Document:
    id: int
    content: str
