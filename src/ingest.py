import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


LOGGER = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def ingest_directory(input_dir: Path) -> Tuple[List[Tuple[Path, str]], List[dict]]:
    parsed = []
    failures = []
    seen_hashes = set()

    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue

        try:
            file_hash = _hash_file(path)
            if file_hash in seen_hashes:
                LOGGER.warning("Skipping duplicate file: %s", path.name)
                continue
            seen_hashes.add(file_hash)

            text = read_pdf(path)
            if not text.strip():
                raise ValueError("PDF contained no extractable text")

            parsed.append((path, text))
        except Exception as exc:
            LOGGER.exception("Failed to read %s", path)
            failures.append({
                "source_file": path.name,
                "error": str(exc),
            })

    return parsed, failures
