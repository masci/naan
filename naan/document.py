from dataclasses import dataclass


@dataclass
class Document:
    idx: int
    content: str
