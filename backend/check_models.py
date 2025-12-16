"""
Quick test: Check which embedding models are already available
Run this to see what's working without downloading anything
"""

from sentence_transformers import SentenceTransformer
import os

LOCAL_MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base",
    "intfloat/multilingual-e5-base",
    "hkunlp/instructor-large",
    "Alibaba-NLP/gte-large-en-v1.5",
    "jinaai/jina-embeddings-v2-base-en"
]

def check_model(model_name):
    """Check if model is available locally"""
    try:
        # Try with local_files_only to avoid downloading
        # Some models need trust_remote_code=True
        if "Alibaba-NLP" in model_name or "gte-large" in model_name:
            model = SentenceTransformer(model_name, local_files_only=True, trust_remote_code=True)
        else:
            model = SentenceTransformer(model_name, local_files_only=True)
        test_embedding = model.encode(["test"], show_progress_bar=False)
        dimension = len(test_embedding[0])
        print(f"✅ {model_name:50} (dim: {dimension})")
        return True
    except Exception:
        print(f"❌ {model_name:50} (not downloaded)")
        return False

def main():
    print("\n" + "="*70)
    print("Checking locally available embedding models...")
    print("="*70 + "\n")
    
    available = []
    missing = []
    
    for model_name in LOCAL_MODELS:
        if check_model(model_name):
            available.append(model_name)
        else:
            missing.append(model_name)
    
    print("\n" + "="*70)
    print(f"Summary: {len(available)}/{len(LOCAL_MODELS)} models available locally")
    print("="*70)
    
    if len(available) == 0:
        print("\n⚠️  NO MODELS FOUND!")
        print("\nYou need to download models first. Run:")
        print("   python download_models.py")
    elif len(missing) > 0:
        print(f"\n⚠️  Missing {len(missing)} models:")
        for m in missing:
            print(f"   - {m}")
        print("\nTo download missing models, run:")
        print("   python download_models.py")
    else:
        print("\n✅ All models available! Backend ready to start.")
    
    print()

if __name__ == "__main__":
    main()
