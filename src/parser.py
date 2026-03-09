import argparse
from pathlib import Path

import fitz


def extract_text(pdf_path: Path) -> str:
    text_chunks = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text().strip()
            if text:
                text_chunks.append(text)
    return "\n\n".join(text_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract plain text from a PDF.")
    parser.add_argument("pdf_path")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).resolve()
    text = extract_text(pdf_path)

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
