"""
extract_qa_corpus.py
════════════════════
Dump the full Q&A corpus from staging MongoDB (both sources).

Sources:
  1. Reviewer dataset  — agriai.questions + agriai.answers
  2. Golden dataset    — golden_db.agri_qa

Output: qa_corpus.jsonl
Each record:
{
  "source":       "reviewer" | "golden",
  "question_id":  str,
  "question":     str,
  "answer":       str,
  "state":        str,
  "crop":         str,
  "author":       str,    # reviewer only
  "metadata":     dict,   # all extra fields
}
"""

import json
import os
import sys
from pathlib import Path

from pymongo import MongoClient
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
import config


def extract_reviewer_qa(db_agriai) -> list:
    """
    Reviewer dataset: questions collection + answers collection.
    questions._id → answers.questionId  (join key)
    """
    print("  Loading reviewer questions ...")
    questions_col = db_agriai[config.STAGING_QA_QUESTIONS]
    answers_col   = db_agriai[config.STAGING_QA_ANSWERS]

    # Build answer lookup: questionId → answer doc
    print("  Building answer lookup ...")
    answer_lookup = {}
    for ans in tqdm(answers_col.find({}), desc="  Answers"):
        qid = str(ans.get("questionId", ans.get("question_id", "")))
        if qid:
            answer_lookup[qid] = ans

    records = []
    for q in tqdm(questions_col.find({}), desc="  Questions"):
        qid    = str(q.get("_id", ""))
        ans    = answer_lookup.get(qid, {})
        details = q.get("details", {})

        records.append({
            "source":       "reviewer",
            "question_id":  qid,
            "question":     q.get("question", q.get("text", "")).strip(),
            "answer":       ans.get("answer", ans.get("text", "")).strip(),
            "state":        details.get("state", q.get("state", "")),
            "crop":         details.get("crop",  q.get("crop",  "")),
            "author":       ans.get("author", ans.get("reviewedBy", "")),
            "metadata": {
                "district":  details.get("district", ""),
                "season":    details.get("season", ""),
                "domain":    details.get("domain", ""),
                "priority":  q.get("priority", ""),
                "source":    q.get("source", ""),
            },
        })

    print(f"  Reviewer Q&A: {len(records):,} records")
    return records


def extract_golden_qa(db_golden) -> list:
    """
    Golden dataset: golden_db.agri_qa
    """
    print("  Loading golden Q&A ...")
    col = db_golden[config.STAGING_GOLDEN_QA]
    records = []

    for doc in tqdm(col.find({}), desc="  Golden Q&A"):
        meta = doc.get("metadata", doc.get("meta_data", {}))

        records.append({
            "source":       "golden",
            "question_id":  str(doc.get("_id", "")),
            "question":     doc.get("question", doc.get("text", "")).strip(),
            "answer":       doc.get("answer", "").strip(),
            "state":        meta.get("State", meta.get("state", "")),
            "crop":         meta.get("Crop",  meta.get("crop",  "")),
            "author":       meta.get("Agri Specialist", meta.get("author", "")),
            "metadata":     meta,
        })

    print(f"  Golden Q&A: {len(records):,} records")
    return records


def extract_qa_corpus():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("Connecting to staging MongoDB ...")
    client = MongoClient(config.STAGING_URI, serverSelectionTimeoutMS=10_000)

    db_agriai  = client[config.STAGING_QA_DB]
    db_golden  = client[config.STAGING_GOLDEN_DB]

    reviewer_records = extract_reviewer_qa(db_agriai)
    golden_records   = extract_golden_qa(db_golden)

    all_records = reviewer_records + golden_records

    # Deduplicate on (question text lowercase)
    seen = set()
    deduped = []
    for r in all_records:
        key = r["question"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f"\nTotal Q&A records : {len(all_records):,}")
    print(f"After dedup       : {len(deduped):,}")

    with open(config.QA_DUMP_JSONL, "w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print(f"Written → {config.QA_DUMP_JSONL}")
    client.close()


if __name__ == "__main__":
    extract_qa_corpus()
