"""
Multi-GPU training for Ajrasakha Embedding Model.
Uses GPU 0 and GPU 1 for maximum throughput.
"""
import os
import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from datasets import Dataset as HFDataset

# ─── Multi-GPU Setup ─────────────────────────────────────
# We use both GPU 0 and GPU 1
# Phase 2: Using the reliable GPU 0 for stability
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ─── Config ──────────────────────────────────────────────
BASE_MODEL       = "models/finetuned/bge-large-agri-multi-gpu"
TRAIN_FILE       = "data/training/train.jsonl"
VAL_FILE         = "data/training/val.jsonl"
OUTPUT_DIR       = "models/finetuned/bge-large-agri-multi-gpu-phase2"
LOGGING_DIR      = "models/finetuned/logs_multi_phase2"

# Optimized Hyperparameters for Dual RTX 6000 (96GB each)
# Phase 2: Polishing phase
NUM_EPOCHS       = 2
TRAIN_BATCH_SIZE = 128     # Reliable single-GPU batch
EVAL_BATCH_SIZE  = 128
LEARNING_RATE    = 1e-5    # Lower LR for polishing
WARMUP_RATIO     = 0.05    # Shorter warmup for Phase 2
SAVE_STEPS       = 100     # Save every 100 steps to catch all minima
EVAL_STEPS       = 100
MAX_SEQ_LENGTH   = 512

def load_triplets(path: str):
    records = {"query": [], "positive": [], "negative": []}
    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} not found.")
        return HFDataset.from_dict(records)
        
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
            records["negative"].append(neg if neg else pos)
    return HFDataset.from_dict(records)

def main():
    print(f"🔧 Loading base model: {BASE_MODEL}")
    # Don't specify a device here; Trainer will handle DataParallel/DDP
    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH

    print(f"📂 Loading training data...")
    train_dataset = load_triplets(TRAIN_FILE)
    val_dataset   = load_triplets(VAL_FILE)
    print(f"  Train: {len(train_dataset):,} triplets")
    print(f"  Val  : {len(val_dataset):,} triplets")

    # MNRL loss with multi-GPU support
    loss = MultipleNegativesRankingLoss(model=model)

    args = SentenceTransformerTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        per_device_eval_batch_size  = EVAL_BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        fp16                        = True,
        batch_sampler               = BatchSamplers.NO_DUPLICATES,
        eval_strategy               = "steps",
        eval_steps                  = EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 20,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        logging_dir                 = LOGGING_DIR,
        logging_steps               = 50,
        report_to                   = "tensorboard",
        run_name                    = "bge-agri-multi-gpu",
        dataloader_num_workers      = 4,
        push_to_hub                 = False,
        gradient_checkpointing      = True,
    )

    trainer = SentenceTransformerTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        loss            = loss,
    )

    print(f"\n🚀 Starting Phase 2: Polishing fine-tuning...")
    print(f"   Base Model            : {BASE_MODEL}")
    print(f"   Total Epochs          : {NUM_EPOCHS}")
    print(f"   Learning Rate         : {LEARNING_RATE}")
    print(f"   Output dir            : {OUTPUT_DIR}\n")

    trainer.train()

    print(f"\n💾 Saving final model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
