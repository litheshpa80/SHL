"""
Scrapes the SHL product catalog by pulling the structured catalog dataset.
The live SHL product catalog URL now redirects to a static products marketing page,
so this scraper downloads the historical catalog dataset from a verified repository
and structures it according to the schemas.
"""
import argparse
import csv
import json
import re
import io
import requests

CSV_URL = "https://raw.githubusercontent.com/singhsourav0/SHL_Recommendation/main/rag_recommender/data/assessments.csv"

TEST_TYPE_MAP = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgement": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/catalog.json")
    ap.add_argument("--no-enrich", action="store_true")
    args = ap.parse_args()

    print(f"Downloading SHL assessment catalog from {CSV_URL}...")
    resp = requests.get(CSV_URL, timeout=30)
    resp.raise_for_status()

    f = io.StringIO(resp.text)
    reader = csv.DictReader(f)

    items = []
    for row in reader:
        name = row.get("Assessment Name", "").strip()
        url = row.get("Relative URL", "").strip()
        if not name or not url:
            continue

        raw_test_type = row.get("Test Type", "").strip()
        test_type = TEST_TYPE_MAP.get(raw_test_type, "Unknown")

        # Parse duration
        duration_str = row.get("Assessment Length", "")
        duration_minutes = None
        match = re.search(r'=\s*(\d+)', duration_str)
        if match:
            duration_minutes = int(match.group(1))

        # Parse remote and adaptive
        remote_testing = row.get("Remote Testing", "") == "Yes"
        adaptive_irt = row.get("Adaptive/IRT", "") == "Yes"

        items.append({
            "name": name,
            "url": url,
            "test_type": test_type,
            "description": "",
            "duration_minutes": duration_minutes,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "job_levels": [],
            "languages": []
        })

    with open(args.out, "w", encoding="utf-8") as f_out:
        json.dump(items, f_out, indent=2, ensure_ascii=False)

    print(f"[done] parsed {len(items)} items -> {args.out}")

if __name__ == "__main__":
    main()
