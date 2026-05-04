
import torch
import os
from sentence_transformers import SentenceTransformer

def soup_models(model_paths, output_path):
    print(f"🥣 Souping {len(model_paths)} models into {output_path}...")
    
    # Load the first model to get the structure
    base_model = SentenceTransformer(model_paths[0])
    soup_state_dict = base_model.state_dict()
    
    # Initialize soup with first model's weights
    for key in soup_state_dict:
        soup_state_dict[key] = soup_state_dict[key].float()
        
    # Add weights from remaining models
    for path in model_paths[1:]:
        print(f"➕ Adding {path}...")
        current_model = SentenceTransformer(path)
        current_state_dict = current_model.state_dict()
        for key in soup_state_dict:
            soup_state_dict[key] += current_state_dict[key].float()
            
    # Divide by number of models to get the average
    num_models = len(model_paths)
    for key in soup_state_dict:
        soup_state_dict[key] = (soup_state_dict[key] / num_models).to(torch.float16)
        
    # Save the souped model
    base_model.save(output_path)
    # Overwrite the weights with our souped ones
    torch.save(soup_state_dict, os.path.join(output_path, "model.safetensors")) # This is a bit tricky with ST
    # Actually, the best way is to load state dict back into model
    base_model.load_state_dict(soup_state_dict)
    base_model.save(output_path)
    print("✅ Soup created successfully!")

if __name__ == "__main__":
    top_checkpoints = [
        "models/finetuned/bge-large-agri-multi-gpu/checkpoint-6400",
        "models/finetuned/bge-large-agri-multi-gpu/checkpoint-6500",
        "models/finetuned/bge-large-agri-multi-gpu-phase2/checkpoint-2000"
    ]
    output_dir = "models/finetuned/bge-large-agri-soup"
    os.makedirs(output_dir, exist_ok=True)
    soup_models(top_checkpoints, output_dir)
