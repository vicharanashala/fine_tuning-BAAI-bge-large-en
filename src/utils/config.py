import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── MongoDB URIs ──────────────────────────────────────────────────────────────
STAGING_URI  = os.getenv("MONGODB_STAGING_URI")
PROD_URI     = os.getenv("MONGODB_PROD_URI")

# ── Staging databases / collections ──────────────────────────────────────────
STAGING_QA_DB         = "agriai"
STAGING_QA_QUESTIONS  = "questions"    # reviewer questions
STAGING_QA_ANSWERS    = "answers"      # reviewer answers

STAGING_GOLDEN_DB     = "golden_db"
STAGING_GOLDEN_QA     = "agri_qa"      # golden Q&A  (12k Q&A source 1)
STAGING_POP           = "pop"          # 31k POP chunks

# ── Production databases / collections ───────────────────────────────────────
PROD_DB               = "test"
PROD_CONVERSATIONS    = "conversations"
PROD_MESSAGES         = "messages"

# ── Tool names to track (partial match — MCP adds suffixes) ──────────────────
TOOL_QA_REVIEWER   = "get_context_from_reviewer_dataset"
TOOL_QA_GOLDEN     = "get_context_from_golden_dataset"
TOOL_POP           = "get_context_from_package_of_practices"

# ── Path Configuration ──────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = DATA_DIR / "processed"
RAW_DIR     = DATA_DIR / "raw"
GOLD_DIR    = DATA_DIR / "gold"
MODELS_DIR  = BASE_DIR / "models"

# Within data/processed/:
LOG_QA_JSONL  = str(OUTPUT_DIR / "logs_qa_retrieval.jsonl")   # A2 raw logs
LOG_POP_JSONL = str(OUTPUT_DIR / "logs_pop_retrieval.jsonl")  # B2 raw logs
QA_CORPUS     = str(OUTPUT_DIR / "qa_corpus.jsonl")           # full Q&A corpus
POP_CORPUS    = str(OUTPUT_DIR / "pop_corpus.jsonl")          # full POP corpus

# Eval split: last N days of logs are locked for evaluation
EVAL_SPLIT_DAYS = 14   # last 2 weeks → eval; everything older → training
