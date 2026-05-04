"""
Step 4: benchmark_models.py
Side-by-side comparison: base BGE-large-en-v1.5 vs fine-tuned model.
  - Runs evaluate_model.py for both
  - Produces a diff table showing improvement per metric
  - Saves a summary report
"""
import os
import json
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from evaluate_model import compute_metrics

# ─── Config ──────────────────────────────────────────────
BASE_MODEL       = "BAAI/bge-large-en-v1.5"
FINETUNED_MODEL  = "models/finetuned/bge-large-agri-v1"
REPORT_DIR       = Path("data/evaluation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "MRR", "NDCG@10", "HitRate@5"]

def print_table(base, ft):
    print(f"\n{'─'*65}")
    print(f"  {'Metric':<14} | {'Base BGE':>12} | {'Fine-tuned':>12} | {'Delta':>10}")
    print(f"{'─'*65}")
    improvements = 0
    for m in METRICS:
        b = base.get(m, 0.0)
        f = ft.get(m, 0.0)
        delta = f - b
        symbol = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        if delta > 0:
            improvements += 1
        print(f"  {m:<14} | {b:>12.4f} | {f:>12.4f} | {delta:>+9.4f} {symbol}")
    print(f"{'─'*65}")
    print(f"  Improved on {improvements}/{len(METRICS)} metrics")
    return improvements

def main():
    print("="*65)
    print("  AJRASAKHA EMBEDDING MODEL BENCHMARK")
    print("="*65)

    print("\n\n[1/2] Evaluating BASE model...")
    base_metrics = compute_metrics(BASE_MODEL, "base_bge_large")

    print("\n\n[2/2] Evaluating FINE-TUNED model...")
    if not Path(FINETUNED_MODEL).exists():
        print(f"⚠️  Fine-tuned model not found at {FINETUNED_MODEL}")
        print("    Run: python src/training/train_embedding.py")
        sys.exit(1)
    ft_metrics = compute_metrics(FINETUNED_MODEL, "finetuned_bge_agri_v1")

    print("\n\n" + "="*65)
    print("  BENCHMARK RESULTS — BASE vs FINE-TUNED")
    improvements = print_table(base_metrics, ft_metrics)

    # Save comparison report
    report = {
        "base":       base_metrics,
        "finetuned":  ft_metrics,
        "delta":      {m: round(ft_metrics.get(m, 0) - base_metrics.get(m, 0), 4) for m in METRICS},
        "n_improved": improvements,
    }
    report_path = REPORT_DIR / "benchmark_comparison.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  📄 Full report saved to: {report_path}")

if __name__ == "__main__":
    main()
