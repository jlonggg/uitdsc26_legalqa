"""Các hàm dựng catalog corpus mà build_catalog.py / index_es.py dùng (load_contexts, build_chunks -> chunk_document
1500/1800 ký tự, write_corpus_catalog, readable_title, lexical_text, title_text). Giữ nguyên logic đã dùng khi sinh
bài nộp để chunk_id / corpus.jsonl trùng byte."""

from __future__ import annotations

import json
import os
import re
import unicodedata
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


class ConfigurationError(RuntimeError):
    """Raised when configuration or a competition invariant is violated."""


@dataclass(frozen=True)
class LegalDocument:
    doc_id: str
    name: str
    link: str
    passage: str


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    doc_id: str
    name: str
    link: str
    article: str
    order: int
    text: str
    normalized: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LegalChunk":
        return cls(
            chunk_id=str(value["chunk_id"]),
            doc_id=str(value["doc_id"]),
            name=str(value.get("name", "")),
            link=str(value.get("link", "")),
            article=str(value.get("article", "")),
            order=int(value.get("order", 0)),
            text=str(value["text"]),
            normalized=str(value.get("normalized") or normalize_text(str(value["text"]))),
        )


def resolve_path(config: Mapping[str, Any], value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(str(config.get("base_dir", Path.cwd()))) / path
    return path.resolve()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).replace("đ", "đ").replace("Đ", "đ")
    return re.sub(r"\s+", " ", text).strip().lower()


_LEGAL_TOKEN_RE = re.compile(
    r"\d+(?:[./-]\d+)+(?:[/-][a-zA-ZĐđ0-9]+)*|[a-zA-ZÀ-ỹĐđ]+(?:[-'][a-zA-ZÀ-ỹĐđ]+)*|\d+",
    re.UNICODE,
)


def legal_tokenize(text: str) -> list[str]:
    """Tokenize consistently while preserving legal document numbers."""
    return _LEGAL_TOKEN_RE.findall(normalize_text(text))


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "d"))
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def title_tokens(text: str) -> list[str]:
    """Tokens for matching a query against a document title.

    Document names are unaccented URL slugs ("...Dieu-le-Hiep-hoi-Dieu-Viet-Nam-208881"),
    but `_LEGAL_TOKEN_RE` keeps a hyphen chain as ONE token and `normalize_text` never
    strips accents, so an accented query shares almost no title tokens with a slug.
    Splitting on hyphens and folding accents makes the two sides comparable. Applied to BOTH index and query text, or the halves stop matching.
    """
    return [tok for tok in re.split(r"[^a-z0-9]+", _strip_accents(normalize_text(text))) if tok]


def lexical_text(text: str) -> str:
    return " ".join(legal_tokenize(text))


_SLUG_TAIL_RE = re.compile(r"-\d{4,}$")


def readable_title(name: str) -> str:
    """Turn a URL slug into something the embedder can read.

    "Quyet-dinh-948-QD-BNV-nam-2013-Dieu-le-Hiep-hoi-Dieu-Viet-Nam-208881"
    -> "Quyet dinh 948 QD BNV nam 2013 Dieu le Hiep hoi Dieu Viet Nam".
    The trailing corpus id is dropped: it is a thuvienphapluat page number, never
    something a question refers to, and it only adds a junk token to the vector.
    """
    return _SLUG_TAIL_RE.sub("", str(name)).replace("-", " ").strip()


def title_text(text: str) -> str:
    return " ".join(title_tokens(text))


def _json_records(value: Any, source: str) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and all(key in value for key in ("id", "passage")):
        return [value]
    if isinstance(value, dict):
        records = list(value.values())
        if all(isinstance(row, dict) for row in records):
            return records
    raise ValueError(f"{source}: expected a context record, list, or object of records")


def _context_members(path: Path) -> Iterator[tuple[str, bytes]]:
    if path.is_dir():
        files = sorted(path.rglob("context_*.json"))
        if not files:
            files = sorted(path.rglob("*.json"))
        for file_path in files:
            yield str(file_path), file_path.read_bytes()
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name for name in archive.namelist() if not name.endswith("/") and Path(name).name.startswith("context_")
                and name.lower().endswith(".json")
            )
            if not names:
                names = sorted(name for name in archive.namelist() if name.lower().endswith(".json"))
            for name in names:
                yield name, archive.read(name)
        return
    if path.suffix.lower() == ".json":
        yield str(path), path.read_bytes()
        return
    raise ValueError("contexts must be a directory, JSON file, or ZIP")


def load_contexts(path: str | os.PathLike[str]) -> list[LegalDocument]:
    context_path = Path(path).resolve()
    if not context_path.exists():
        raise FileNotFoundError(context_path)
    documents: list[LegalDocument] = []
    seen: set[str] = set()
    for source, payload in _context_members(context_path):
        try:
            value = json.loads(payload.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid UTF-8 JSON in {source}: {error}") from error
        for row in _json_records(value, source):
            # BTC corpus: 13% of records omit "name" and 20 have an empty passage.
            # Only id/passage carry retrieval signal, so name/link default to "".
            missing = {key for key in ("id", "passage") if key not in row}
            if missing:
                raise ValueError(f"{source}: context record missing {sorted(missing)}")
            doc_id = str(row["id"]).strip()
            passage = str(row["passage"]).strip()
            if not doc_id:
                raise ValueError(f"{source}: id must be non-empty")
            if not passage:
                continue  # empty passage: nothing to index, skip instead of failing
            if doc_id in seen:
                raise ValueError(f"duplicate document id: {doc_id}")
            seen.add(doc_id)
            documents.append(
                LegalDocument(doc_id, str(row.get("name") or ""), str(row.get("link") or ""), passage)
            )
    if not documents:
        raise ValueError(f"no context records found under {context_path}")
    return documents


_BOUNDARY_RE = re.compile(r"(?im)(?=^[ \t]*(?:chương|mục|tiểu mục|điều)\s+[\divxlc]+\b[^\n]*)")


_ARTICLE_RE = re.compile(r"(?im)^\s*(điều\s+\d+[a-zđ]?(?:[.-]\d+)?)\b")


_CLAUSE_RE = re.compile(r"(?m)(?=^\s*(?:\d+[.)]|[a-zđ][.)]|[-–•])\s+)")


def _hard_split(text: str, maximum: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > maximum:
        cut = remaining.rfind(" ", 0, maximum + 1)
        if cut < maximum // 2:
            cut = maximum
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _pack_units(units: Sequence[str], target: int, maximum: int) -> list[str]:
    expanded = [piece for unit in units for piece in _hard_split(unit, maximum) if piece]
    packed: list[str] = []
    current = ""
    for unit in expanded:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if current and len(candidate) > maximum:
            packed.append(current)
            current = unit
        else:
            current = candidate
        if len(current) >= target:
            packed.append(current)
            current = ""
    if current:
        if packed and len(packed[-1]) + len(current) + 1 <= maximum:
            packed[-1] = f"{packed[-1]}\n{current}"
        else:
            packed.append(current)
    return packed


def chunk_document(
    doc_id: str,
    name: str,
    link: str,
    passage: str,
    target_chars: int = 1500,
    max_chars: int = 1800,
) -> list[LegalChunk]:
    if target_chars <= 0 or max_chars < target_chars:
        raise ConfigurationError("chunk target/max must satisfy 0 < target <= max")
    normalized_newlines = unicodedata.normalize("NFKC", passage).replace("\r\n", "\n").replace("\r", "\n").strip()
    sections = [section.strip() for section in _BOUNDARY_RE.split(normalized_newlines) if section.strip()]
    if not sections:
        sections = [normalized_newlines]
    chunk_texts: list[tuple[str, str]] = []
    current_article = ""
    for section in sections:
        match = _ARTICLE_RE.search(section)
        if match:
            current_article = match.group(1).strip()
        units = [unit.strip() for unit in _CLAUSE_RE.split(section) if unit.strip()]
        if len(units) == 1:
            units = [unit.strip() for unit in re.split(r"\n\s*\n+", section) if unit.strip()]
        for packed in _pack_units(units, target_chars, max_chars):
            chunk_texts.append((current_article, packed))
    result: list[LegalChunk] = []
    namespace = uuid.UUID("7f1123cb-585d-4f7a-bab5-39a90c428314")
    for order, (article, text) in enumerate(chunk_texts):
        chunk_id = str(uuid.uuid5(namespace, f"{doc_id}\n{order}\n{text}"))
        result.append(LegalChunk(chunk_id, doc_id, name, link, article, order, text, normalize_text(text)))
    return result


def build_chunks(documents: Sequence[LegalDocument], config: Mapping[str, Any]) -> list[LegalChunk]:
    settings = config.get("chunking", {})
    target = int(settings.get("target_chars", 1500))
    maximum = int(settings.get("max_chars", 1800))
    return [chunk for document in documents for chunk in chunk_document(document.doc_id, document.name, document.link, document.passage, target, maximum)]


def _artifact_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    # "artifacts" = output dir passed in by build_catalog.py ($WORK/retrieval), not the repo's artifacts/
    paths = config.get("paths", {})
    artifacts = resolve_path(config, paths.get("artifacts", "artifacts"))
    catalog = artifacts / "corpus.jsonl"
    manifest = artifacts / "prepare_manifest.json"
    return artifacts, catalog, manifest


def write_corpus_catalog(
    documents: Sequence[LegalDocument], chunks: Sequence[LegalChunk], config: Mapping[str, Any]
) -> None:
    artifacts, catalog, _ = _artifact_paths(config)
    artifacts.mkdir(parents=True, exist_ok=True)
    # Per-process temp name: concurrent shards each write their own file and rename
    # onto the same target. A shared ".jsonl.tmp" made the second rename fail with
    # ENOENT once the first shard had already moved it away.
    temporary = catalog.with_suffix(f".jsonl.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(canonical_json({"type": "meta", "documents": [asdict(item) for item in documents]}) + "\n")
        for chunk in chunks:
            stream.write(canonical_json({"type": "chunk", **asdict(chunk)}) + "\n")
    temporary.replace(catalog)
