from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
cache_dir = "./model_cache"  

print(f"Downloading model {model_name} to {cache_dir}...")
model = SentenceTransformer(model_name, cache_folder=cache_dir)
print("Model download complete!")
