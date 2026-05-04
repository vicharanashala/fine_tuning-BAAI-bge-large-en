
import torch
import os
from sentence_transformers import SentenceTransformer

def fuse_models(model_a_path, model_b_path, alpha, output_path):
    """
    Fused = (1 - alpha) * Model_A + alpha * Model_B
    Alpha 1.0 = All Model B
    Alpha 0.0 = All Model A
    """
    print(f"🧬 Fusing {model_a_path} and {model_b_path} (Alpha={alpha}) -> {output_path}")
    
    model_a = SentenceTransformer(model_a_path)
    model_b = SentenceTransformer(model_b_path)
    
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    
    fused_state = {}
    for key in state_a:
        fused_state[key] = ((1 - alpha) * state_a[key].float() + alpha * state_b[key].float()).to(torch.float16)
        
    model_a.load_state_dict(fused_state)
    model_a.save(output_path)
    print("✅ Fusion complete!")

if __name__ == "__main__":
    base = "BAAI/bge-large-en-v1.5"
    specialist = "models/finetuned/bge-large-agri-multi-gpu/checkpoint-6400"
    
    # The Micro-Sweep for Global Optimum
    fuses = [0.02, 0.05, 0.08, 0.12, 0.15]
    for alpha in fuses:
        out_dir = f"models/finetuned/bge-large-agri-fused-{int(alpha*100)}"
        fuse_models(base, specialist, alpha, out_dir)
