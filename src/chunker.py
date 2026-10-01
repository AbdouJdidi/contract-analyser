# src/chunker.py
import re

SECTION_RE = re.compile(
    r"^[ \t]*(?P<num>\d+(?:\.\d+)*)[ \t.]+(?P<title>[A-Z][^\n]{2,90})$",
    re.MULTILINE,
)
HEADING_RE = re.compile(r"^[ \t]*(?P<title>[A-Z][A-Z \-&',\.]{4,70})[ \t]*$", re.MULTILINE)
PAGE_RE = re.compile(r"Page\s*-\s*\d+\s*-")

MAX_CHARS = 1800      
OVERLAP   = 200

def normalize(s: str) -> str:
    s = PAGE_RE.sub(" ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def find_boundaries(text: str):
    """Positions where a new section starts, with its label."""
    marks = []
    for m in SECTION_RE.finditer(text):
        marks.append((m.start(), f"{m.group('num')} {m.group('title').strip()}"))
    for m in HEADING_RE.finditer(text):
        marks.append((m.start(), m.group("title").strip()))
    marks.sort()
    # drop duplicates at the same position
    out, seen = [], set()
    for pos, label in marks:
        if pos not in seen:
            seen.add(pos)
            out.append((pos, label))
    return out

def split_long(start: int, end: int, text: str):
    """Fallback: cut an oversized block on paragraph breaks, with overlap."""
    pieces, cur = [], start
    while cur < end:
        stop = min(cur + MAX_CHARS, end)
        if stop < end:
            br = text.rfind("\n\n", cur + MAX_CHARS // 2, stop)
            if br != -1:
                stop = br
        pieces.append((cur, stop))
        cur = max(stop - OVERLAP, stop)
    return pieces

def chunk_contract(text: str, title: str):
    clean_title = re.sub(r"[_\-]+", " ", title)
    clean_title = re.sub(r"\b(EX|S|\d{6,}|\d{2} \d{2} \d{4}|\d{8})\b", " ", clean_title)
    clean_title = re.sub(r"\s+", " ", clean_title).strip()[:60]
    marks = find_boundaries(text)
    if not marks:
        marks = [(0, "PREAMBLE")]
    if marks[0][0] > 0:
        marks.insert(0, (0, "PREAMBLE"))

    chunks = []
    for i, (pos, label) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        if end - pos < 40:                     
            continue
        for s, e in split_long(pos, end, text):
            body = normalize(text[s:e])
            if len(body) < 40:
                continue
            chunks.append({
                "doc": title,
                "section": label,
                "start": s,
                "end": e,
                "text": f"[{title} — {label}]\n{body}",
            })
    return chunks