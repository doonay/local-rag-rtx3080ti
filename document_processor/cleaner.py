import re
import unicodedata


_PAGE_MARKER = re.compile(r"^\s*(?:page|страница)\s*\d+(?:\s*(?:of|из)\s*\d+)?\s*$", re.IGNORECASE)
_BARE_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")
_RULE = re.compile(r"^[\-=*_]{5,}$")


def clean_text(text: str) -> str:
    """Normalize extracted text without discarding meaningful paragraphs."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if _PAGE_MARKER.match(line) or _BARE_PAGE_NUMBER.match(line) or _RULE.match(line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
