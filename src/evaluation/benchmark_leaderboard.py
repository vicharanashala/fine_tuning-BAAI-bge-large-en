
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

def calculate_metrics(query_embeddings, corpus_embeddings, positive_indices):
    """Calculate NDCG@10, Recall@1, and MRR."""
    # Compute cosine similarity
    cos_scores = util.cos_sim(query_embeddings, corpus_embeddings)
    
    # Sort scores
    top_k_indices = torch.topk(cos_scores, k=10, dim=1).indices.cpu().numpy()
    
    recall_at_1 = 0
    mrr = 0
    ndcg = 0
    
    for i, top_k in enumerate(top_k_indices):
        target = positive_indices[i]
        
        # Recall@1
        if top_k[0] == target:
            recall_at_1 += 1
            
        # MRR
        for rank, idx in enumerate(top_k):
            if idx == target:
                mrr += 1 / (rank + 1)
                ndcg += 1 / np.log2(rank + 2)
                break
                
    return {
        "Recall@1": recall_at_1 / len(query_embeddings),
        "MRR": mrr / len(query_embeddings),
        "NDCG@10": ndcg / len(query_embeddings)
    }

def evaluate_model(model_path, eval_file):
    print(f"🔍 Evaluating: {model_path}")
    model = SentenceTransformer(model_path)
    
    queries = []
    positives = []
    negatives = []
    
    with open(eval_file) as f:
        for line in f:
            d = json.loads(line)
            queries.append(d["query"])
            positives.append(d["positive"])
            negatives.append(d["negative"])
            
    # Combine positives and negatives for a unique corpus
    corpus = list(set(positives + negatives))
    corpus_map = {text: i for i, text in enumerate(corpus)}
    positive_indices = [corpus_map[p] for p in positives]
    
    # Encode
    query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
    
    metrics = calculate_metrics(query_embeddings, corpus_embeddings, positive_indices)
    return metrics

if __name__ == "__main__":
    eval_file = "data/training/eval_locked.jsonl"
    
    # The Micro-Sweep Tournament
    finalists = [
        "BAAI/bge-large-en-v1.5", 
        "models/finetuned/bge-large-agri-fused-2",
        "models/finetuned/bge-large-agri-fused-5",
        "models/finetuned/bge-large-agri-fused-8",
        "models/finetuned/bge-large-agri-fused-10",
        "models/finetuned/bge-large-agri-fused-12",
        "models/finetuned/bge-large-agri-fused-15"
    ]
    
    results = {}
    for path in finalists:
        try:
            results[path] = evaluate_model(path, eval_file)
            print(f"Results for {path}: {results[path]}")
        except Exception as e:
            print(f"❌ Failed to evaluate {path}: {e}")
            
    # Print Table
    print("\n" + "="*50)
    print(f"{'Model Path':<60} | {'NDCG@10':<10} | {'Recall@1':<10}")
    print("-" * 85)
    for path, metrics in results.items():
        print(f"{path:<60} | {metrics['NDCG@10']:.4f} | {metrics['Recall@1']:.4f}")
