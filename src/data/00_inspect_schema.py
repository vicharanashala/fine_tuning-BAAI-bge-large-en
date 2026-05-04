"""
inspect_pop_schema.py
══════════════════════
One-off script to print sample POP and log documents so you can verify
field names before running the full extraction.

Run this FIRST to confirm field names match what the extraction scripts expect.
"""

import json
import sys
from pathlib import Path
from pprint import pprint

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
import config


def inspect(n_samples: int = 3):
    print("=" * 60)
    print("STAGING — POP chunk sample")
    print("=" * 60)
    client_s = MongoClient(config.STAGING_URI, serverSelectionTimeoutMS=10_000)
    pop_col  = client_s[config.STAGING_GOLDEN_DB][config.STAGING_POP]
    for doc in pop_col.find({}).limit(n_samples):
        print("\nKeys:", list(doc.keys()))
        print("metadata keys:", list(doc.get("metadata", {}).keys()))
        # show text field candidates
        for key in ["text", "content", "chunk_text", "page_content"]:
            val = doc.get(key, "")
            if val:
                print(f"  content field '{key}': {str(val)[:200]}")
        print("---")
    client_s.close()

    print("\n" + "=" * 60)
    print("PRODUCTION — sample LLM message with tool calls")
    print("=" * 60)
    client_p = MongoClient(config.PROD_URI, serverSelectionTimeoutMS=10_000)
    msg_col  = client_p[config.PROD_DB][config.PROD_MESSAGES]

    # Find one message that has tool calls
    for doc in msg_col.find({"isCreatedByUser": False, "content": {"$exists": True}}).limit(50):
        content = doc.get("content", [])
        tool_calls = [c for c in content if isinstance(c, dict) and c.get("type") == "tool_call"]
        if not tool_calls:
            continue

        # Check if any is a target tool
        for tc in tool_calls:
            name = tc.get("tool_call", {}).get("name", "")
            if any(t in name for t in [config.TOOL_QA_REVIEWER, config.TOOL_QA_GOLDEN, config.TOOL_POP]):
                print(f"\nConversation: {doc.get('conversationId')}")
                print(f"Tool name   : {name}")
                print(f"Args        : {json.dumps(tc.get('tool_call', {}).get('args', {}), indent=2)}")
                output = tc.get("output", [])
                print(f"Output type : {type(output)}")
                if isinstance(output, list) and output:
                    first = output[0]
                    print(f"Output[0]   : type={first.get('type')}")
                    text = first.get("text", "")
                    try:
                        parsed = json.loads(text) if isinstance(text, str) else text
                        print(f"Output parsed (first item): {json.dumps(parsed[0] if isinstance(parsed, list) else parsed, indent=2)[:800]}")
                    except Exception as e:
                        print(f"Output text (raw): {str(text)[:400]}")
                break
        else:
            continue
        break

    client_p.close()


if __name__ == "__main__":
    inspect()
