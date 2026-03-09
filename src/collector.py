import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv
from pyalex import Works, config

load_dotenv()

# OpenAlex polite pool
config.email = os.getenv("OPENALEX_EMAIL") or os.getenv("EMAIL") or ""

DEFAULT_QUERY = "large language models"
DEFAULT_LIMIT = 10
MAX_PDF_MB = 25
REQUEST_TIMEOUT = 20


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "paper"


def _download_pdf(url: str, out_path: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            content_type = r.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                return False

            max_bytes = MAX_PDF_MB * 1024 * 1024
            total = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        return False
                    f.write(chunk)
        return True
    except Exception:
        return False


def _pick_pdf_url(work: Dict) -> Optional[str]:
    best = work.get("best_oa_location") or {}
    pdf_url = best.get("pdf_url")
    if pdf_url:
        return pdf_url

    # Fallback to OA landing if it is a PDF
    oa_url = (work.get("open_access") or {}).get("oa_url")
    if oa_url and oa_url.lower().endswith(".pdf"):
        return oa_url
    return None


def download_papers(query: str, limit: int, data_dir: Path) -> Path:
    raw_dir = data_dir / "raw_pdfs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Dict] = {}

    results = (
        Works()
        .search(query)
        .filter(is_oa=True)
        .sort("cited_by_count:desc")
        .get(page=1, per_page=limit)
    )

    for i, work in enumerate(results, start=1):
        pdf_url = _pick_pdf_url(work)
        if not pdf_url:
            continue

        title = work.get("title") or "untitled"
        year = work.get("publication_year")
        doi = work.get("doi")
        work_id = work.get("id") or ""

        base_name = _slug(f"{i}-{title}")[:80]
        file_name = f"{base_name}.pdf"
        file_path = raw_dir / file_name

        print(f"Downloading {i}/{limit}: {title}")
        ok = _download_pdf(pdf_url, file_path)
        if not ok:
            if file_path.exists():
                file_path.unlink()
            continue

        metadata[file_name] = {
            "title": title,
            "year": year,
            "doi": doi,
            "source": "openalex",
            "work_id": work_id,
            "pdf_url": pdf_url,
        }

        time.sleep(0.2)

    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download open-access PDFs using OpenAlex.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = (base_dir / args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if not config.email:
        print("Warning: OPENALEX_EMAIL not set. Please set it for polite pool access.")

    metadata_path = download_papers(args.query, args.limit, data_dir)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
