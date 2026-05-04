"""
extract_logs.py
═══════════════
Extract Q&A and POP retrieval logs from Ajrasakha production MongoDB.

For each conversation:
  1. Find the farmer's original query (user message)
  2. Find LLM message tool calls for:
       - get_context_from_reviewer_dataset   → A2 (Q&A logs)
       - get_context_from_golden_dataset     → A2 (Q&A logs)
       - get_context_from_package_of_practices → B2 (POP logs)
  3. Parse the tool args (query sent to retriever) and output (retrieved items)
  4. Write one JSONL record per tool call

Output format per record:
{
  "conversation_id": str,
  "split": "eval" | "train",                   # based on EVAL_SPLIT_DAYS
  "original_farmer_query": str,                 # raw text from user message
  "tool_name": str,                             # which retrieval tool was called
  "tool_query": str,                            # query the LLM sent to the tool
  "meta": {state, crop, state_code, ...},       # tool args extras
  "retrieved": [                                # parsed tool output
    {
      "rank": int,
      # Q&A:  question_id, question_text, answer_text, similarity_score, author
      # POP:  chunk_id, content, metadata, similarity_score
    },
    ...
  ],
  "timestamp": ISO str,
  "message_id": str
}
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from pymongo import MongoClient
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
import config

# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_tool_output(output_field) -> list:
    """
    Tool output in messages is stored as:
      [{"type": "text", "text": "<JSON string>"}]

    We want the parsed JSON value of .text.
    Always returns a list. If parsing fails, returns [{"raw_text_error": text}].
    """
    if not output_field:
        return []
    try:
        # output_field can be a list or a JSON string
        if isinstance(output_field, str):
            output_field = json.loads(output_field)

        for item in output_field:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    try:
                        parsed = json.loads(text)
                        return parsed if isinstance(parsed, list) else [parsed]
                    except json.JSONDecodeError:
                        return [{"raw_text_error": text}]
                elif isinstance(text, (list, dict)):
                    return text if isinstance(text, list) else [text]
    except Exception:
        pass
    return []


def _parse_tool_args(args_field) -> dict:
    """
    args can be a dict already or a JSON string.
    """
    if isinstance(args_field, dict):
        return args_field
    if isinstance(args_field, str):
        try:
            return json.loads(args_field)
        except json.JSONDecodeError:
            return {}
    return {}


def _is_target_tool(tool_name: str) -> tuple[bool, str]:
    """
    Returns (is_target, canonical_tool_key).
    Handles MCP suffix patterns: tool_base_name_mcp_<server>
    """
    name = tool_name or ""
    if config.TOOL_QA_REVIEWER in name:
        return True, "qa_reviewer"
    if config.TOOL_QA_GOLDEN in name:
        return True, "qa_golden"
    if config.TOOL_POP in name:
        return True, "pop"
    return False, ""


def _eval_or_train(timestamp: datetime, cutoff: datetime) -> str:
    if timestamp and timestamp >= cutoff:
        return "eval"
    return "train"


# ── main extraction ───────────────────────────────────────────────────────────


def extract_logs_from_uri(uri: str, env_name: str, qa_records: list, pop_records: list, limit: int = None):
    print(f"\nConnecting to {env_name} MongoDB ...")
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    db = client[config.PROD_DB]
    conv_col = db[config.PROD_CONVERSATIONS]
    msg_col  = db[config.PROD_MESSAGES]

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=config.EVAL_SPLIT_DAYS)
    
    # ── Step 1: Build user-message lookup per conversation ────────────────────
    print(f"[{env_name}] Loading user messages ...")
    user_msg_by_conv = {}   # conversationId → {parentMessageId, text, createdAt}

    user_cursor = msg_col.find(
        {"isCreatedByUser": True, "text": {"$exists": True, "$ne": ""}},
        {"conversationId": 1, "text": 1, "messageId": 1, "createdAt": 1}
    )
    for doc in tqdm(user_cursor, desc=f"[{env_name}] User messages"):
        cid = doc.get("conversationId")
        if cid:
            existing = user_msg_by_conv.get(cid)
            ts = doc.get("createdAt")
            if existing is None or (ts and ts < existing["createdAt"]):
                user_msg_by_conv[cid] = {
                    "original_farmer_query": doc.get("text", "").strip(),
                    "createdAt": ts,
                }

    print(f"[{env_name}] Found {len(user_msg_by_conv):,} unique conversations with user messages")

    # ── Step 2: Scan LLM messages for target tool calls ───────────────────────
    print(f"[{env_name}] Scanning LLM messages for retrieval tool calls ...")

    llm_filter = {"isCreatedByUser": False, "content": {"$exists": True}}
    if limit:
        llm_cursor = msg_col.find(llm_filter).limit(limit)
    else:
        llm_cursor = msg_col.find(llm_filter)

    total_tool_calls = 0
    total_messages   = 0

    for msg_doc in tqdm(llm_cursor, desc=f"[{env_name}] LLM messages"):
        total_messages += 1
        cid        = msg_doc.get("conversationId", "")
        message_id = str(msg_doc.get("_id", ""))
        timestamp  = msg_doc.get("createdAt")

        if timestamp and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        content = msg_doc.get("content", [])
        if not isinstance(content, list):
            continue

        user_info = user_msg_by_conv.get(cid, {})
        original_query = user_info.get("original_farmer_query", "")

        for item in content:
            if not isinstance(item, dict): continue
            if item.get("type") != "tool_call": continue

            tool_call = item.get("tool_call", {})
            if not isinstance(tool_call, dict): continue

            tool_name = tool_call.get("name", "")
            is_target, tool_key = _is_target_tool(tool_name)
            if not is_target:
                continue

            total_tool_calls += 1
            args    = _parse_tool_args(tool_call.get("args", {}))
            results = _parse_tool_output(tool_call.get("output", []))
            split   = _eval_or_train(timestamp, cutoff_date)

            record = {
                "conversation_id":      cid,
                "message_id":           message_id,
                "env":                  env_name,
                "split":                split,
                "original_farmer_query": original_query,
                "tool_name":            tool_key,
                "tool_raw_name":        tool_name,
                "timestamp":            timestamp.isoformat() if timestamp else None,
            }

            if tool_key in ("qa_reviewer", "qa_golden"):
                record["tool_query"] = args.get("query", "")
                record["meta"] = {"state": args.get("state", ""), "crop": args.get("crop", "")}
                record["retrieved"] = []
                for i, r in enumerate(results):
                    if not isinstance(r, dict): continue
                    if "raw_text_error" in r:
                        record["retrieved"].append({"rank": i + 1, "raw_text_error": r["raw_text_error"]})
                    else:
                        record["retrieved"].append({
                            "rank":             i + 1,
                            "question_id":      r.get("question_id", r.get("id", "")),
                            "question_text":    r.get("question_text", r.get("question", "")),
                            "answer_text":      r.get("answer_text", r.get("answer", "")),
                            "author":           r.get("author", r.get("meta_data", {}).get("Agri Specialist", "")),
                            "similarity_score": r.get("similarity_score", r.get("meta_data", {}).get("similarity_score", None)),
                        })
                qa_records.append(record)

            elif tool_key == "pop":
                record["tool_query"] = args.get("query", "")
                record["meta"] = {"state_code": args.get("state_code", "")}
                record["retrieved"] = []
                for i, r in enumerate(results):
                    if not isinstance(r, dict): continue
                    if "raw_text_error" in r:
                        record["retrieved"].append({"rank": i + 1, "raw_text_error": r["raw_text_error"]})
                    else:
                        record["retrieved"].append({
                            "rank":             i + 1,
                            "chunk_id":         r.get("id", r.get("chunk_id", r.get("node_id", ""))),
                            "content":          r.get("text", r.get("content", r.get("chunk_text", ""))),
                            "metadata":         r.get("metadata", {}),
                            "similarity_score": r.get("score", r.get("similarity_score", None)),
                        })
                pop_records.append(record)

    print(f"\n  [{env_name}] Processed {total_messages:,} LLM messages")
    print(f"  [{env_name}] Found {total_tool_calls:,} relevant tool calls")
    client.close()

def extract_logs(limit: int = None):
    """
    Main extraction function.
    limit: optional cap on number of conversations (for testing).
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    qa_records  = []
    pop_records = []

    # Extract from both environments
    extract_logs_from_uri(config.PROD_URI, "production", qa_records, pop_records, limit)
    extract_logs_from_uri(config.STAGING_URI, "staging", qa_records, pop_records, limit)

    print(f"\nTotal Q&A log records : {len(qa_records):,}")
    print(f"Total POP log records : {len(pop_records):,}")

    # ── Step 3: Write outputs ─────────────────────────────────────────────────
    _write_jsonl(qa_records,  config.LOG_QA_JSONL)
    _write_jsonl(pop_records, config.LOG_POP_JSONL)

    # ── Summary ───────────────────────────────────────────────────────────────
    qa_eval  = sum(1 for r in qa_records  if r["split"] == "eval")
    qa_train = sum(1 for r in qa_records  if r["split"] == "train")
    pop_eval  = sum(1 for r in pop_records if r["split"] == "eval")
    pop_train = sum(1 for r in pop_records if r["split"] == "train")

    print(f"\n{'─'*50}")
    print(f"Q&A logs  → eval: {qa_eval:>5,}  |  train: {qa_train:>5,}")
    print(f"POP logs  → eval: {pop_eval:>5,}  |  train: {pop_train:>5,}")
    print(f"\nOutputs written to: {config.OUTPUT_DIR}/")
    print(f"  {config.LOG_QA_JSONL}")
    print(f"  {config.LOG_POP_JSONL}")


def _write_jsonl(records: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"  Wrote {len(records):,} records → {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract Ajrasakha retrieval logs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of LLM messages to scan (for testing)")
    args = parser.parse_args()
    extract_logs(limit=args.limit)
