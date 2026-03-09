import os
import requests
import json
from pyalex import Works, config
from dotenv import load_dotenv

load_dotenv()
config.email = os.getenv("EMAIL")

def download_papers(query, limit=5):
    if not os.path.exists("data/raw_pdfs"):
        os.makedirs("data/raw_pdfs")

    # Ищем статьи с открытым доступом и PDF
    results = Works().search(query).filter(is_oa=True).get(page=1, per_page=limit)
    
    metadata = {}

    for i, work in enumerate(results):
        pdf_url = work.get("open_access", {}).get("oa_url")
        if not pdf_url: continue
        
        file_name = f"paper_{i}.pdf"
        file_path = os.path.join("data/raw_pdfs", file_name)
        
        print(f"Downloading: {work['title']}")
        try:
            response = requests.get(pdf_url, timeout=15)
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            metadata[file_name] = {
                "title": work["title"],
                "doi": work["doi"],
                "url": work["doi"],
                "year": work["publication_year"]
            }
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")

    with open("data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    download_papers("Large Language Models", limit=5)