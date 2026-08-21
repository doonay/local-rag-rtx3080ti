import re
from typing import Any, Iterable


class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, strategy: str = "recursive") -> list[dict[str, Any]]:
        text = text.strip()
        if not text:
            return []
        if strategy == "sentence":
            raw_chunks = self._group_units(re.split(r"(?<=[.!?…])\s+", text))
        elif strategy == "paragraph":
            raw_chunks = self._group_units(re.split(r"\n\s*\n", text))
        elif strategy == "recursive":
            raw_chunks = self._chunk_recursive(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        return [
            {"text": chunk, "index": index, "length": len(chunk), "type": strategy}
            for index, chunk in enumerate(raw_chunks)
            if chunk
        ]

    def _chunk_recursive(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            maximum_end = min(start + self.chunk_size, len(text))
            end = maximum_end
            if maximum_end < len(text):
                search_from = start + max(1, self.chunk_size // 2)
                candidates = [
                    text.rfind(separator, search_from, maximum_end)
                    for separator in ("\n\n", "\n", ". ", "! ", "? ", " ")
                ]
                boundary = max(candidates)
                if boundary > start:
                    end = boundary + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            next_start = end - self.overlap
            start = max(start + 1, next_start)
        return chunks

    def _group_units(self, units: Iterable[str]) -> list[str]:
        normalized = [unit.strip() for unit in units if unit and unit.strip()]
        if not normalized:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_length = 0
        for unit in normalized:
            if len(unit) > self.chunk_size:
                if current:
                    chunks.append("\n\n".join(current))
                    current, current_length = [], 0
                chunks.extend(self._chunk_recursive(unit))
                continue
            additional = len(unit) + (2 if current else 0)
            if current and current_length + additional > self.chunk_size:
                chunks.append("\n\n".join(current))
                overlap_units: list[str] = []
                overlap_length = 0
                for previous in reversed(current):
                    if overlap_length + len(previous) > self.overlap:
                        break
                    overlap_units.insert(0, previous)
                    overlap_length += len(previous) + 2
                current = overlap_units
                current_length = len("\n\n".join(current))
            current.append(unit)
            current_length += len(unit) + (2 if len(current) > 1 else 0)
        if current:
            chunks.append("\n\n".join(current))
        return chunks


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Compatibility helper returning only chunk texts."""
    return [
        chunk["text"]
        for chunk in DocumentChunker(chunk_size=chunk_size, overlap=overlap).chunk_text(text)
    ]
