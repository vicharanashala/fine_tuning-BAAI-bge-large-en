"""
Step 2: train_embedding.py
Fine-tunes BAAI/bge-large-en-v1.5 on the agricultural triplet dataset.

Strategy:
  - Loss: MultipleNegativesRankingLoss (best for retrieval tasks)
  - Negatives: hard negatives from the blueprint (not random in-batch)
  - GPU: Explicitly set to CUDA:0 only (GPU 0)
  - Checkpointing: every 500 steps
  - WandB: optional (disabled if not configured)
"""
import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from datasets import Dataset as HFDataset

# ─── Enforce GPU 0 only ───────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ─── Config ──────────────────────────────────────────────
BASE_MODEL       = "BAAI/bge-large-en-v1.5"
TRAIN_FILE       = "data/training/train.jsonl"
VAL_FILE         = "data/training/val.jsonl"
OUTPUT_DIR       = "models/finetuned/bge-large-agri-v1"
LOGGING_DIR      = "models/finetuned/logs"

# Training hyperparameters
NUM_EPOCHS       = 3
TRAIN_BATCH_SIZE = 32      # per-device; gradient accumulation will scale it
EVAL_BATCH_SIZE  = 64
LEARNING_RATE    = 2e-5
WARMUP_RATIO     = 0.1
SAVE_STEPS       = 500
EVAL_STEPS       = 500
MAX_SEQ_LENGTH   = 512     # BGE-large max

# ─── Load Data ───────────────────────────────────────────

def load_triplets(path: str):
    records = {"query": [], "positive": [], "negative": []}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            q   = d.get("query", "").strip()
            pos = d.get("positive", "").strip()
            neg = d.get("negative", "").strip()
            if not q or not pos:
                continue
            records["query"].append(q)
            records["positive"].append(pos)
            records["negative"].append(neg if neg else pos)  # fallback: pos=neg (MNRL handles it)
    return HFDataset.from_dict(records)

def main():
    print(f"🔧 Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL, device="cuda:0")
    model.max_seq_length = MAX_SEQ_LENGTH

    print(f"📂 Loading training data from {TRAIN_FILE}")
    train_dataset = load_triplets(TRAIN_FILE)
    val_dataset   = load_triplets(VAL_FILE)
    print(f"  Train: {len(train_dataset):,} triplets")
    print(f"  Val  : {len(val_dataset):,} triplets")

    # Loss: MNRL is the gold standard for embedding fine-tuning
    # It treats all other positives in the batch as negatives (in-batch negatives)
    # PLUS uses our explicit hard negatives — best of both worlds
    loss = MultipleNegativesRankingLoss(model=model)

    args = SentenceTransformerTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        per_device_eval_batch_size  = EVAL_BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        fp16                        = True,          # Mixed precision on RTX Pro 6000
        bf16                        = False,
        batch_sampler               = BatchSamplers.NO_DUPLICATES,
        eval_strategy               = "steps",
        eval_steps                  = EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        logging_dir                 = LOGGING_DIR,
        logging_steps               = 100,
        report_to                   = "none",        # set to "wandb" if configured
        run_name                    = "bge-agri-finetune-v1",
        dataloader_num_workers      = 4,
    )

    trainer = SentenceTransformerTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        loss            = loss,
    )

    print(f"\n🚀 Starting fine-tuning on GPU 0...")
    print(f"   Epochs         : {NUM_EPOCHS}")
    print(f"   Batch size     : {TRAIN_BATCH_SIZE}")
    print(f"   Learning rate  : {LEARNING_RATE}")
    print(f"   Max seq length : {MAX_SEQ_LENGTH}")
    print(f"   Output dir     : {OUTPUT_DIR}\n")

    trainer.train()

    print(f"\n💾 Saving final model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
