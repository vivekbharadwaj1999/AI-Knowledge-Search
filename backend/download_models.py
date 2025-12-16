"""
Script to download and verify all local embedding models for VivBot
Run this ONCE to download all models before starting the backend
"""

from sentence_transformers import SentenceTransformer
import sys

# All local embedding models from your config
LOCAL_MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",  # Note: Full path needed
    "intfloat/e5-base",
    "intfloat/multilingual-e5-base", 
    "hkunlp/instructor-large",
    "Alibaba-NLP/gte-large-en-v1.5",
    "jinaai/jina-embeddings-v2-base-en"
]

def download_model(model_name):
    """Download and verify a single model"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Some models (like Alibaba GTE) need trust_remote_code=True
        model = SentenceTransformer(model_name, trust_remote_code=True)
        # Test encode to ensure it works
        test_embedding = model.encode(["test sentence"])
        dimension = len(test_embedding[0])
        print(f"✅ SUCCESS: {model_name} (dimension: {dimension})")
        return True
    except Exception as e:
        print(f"❌ FAILED: {model_name}")
        print(f"   Error: {str(e)[:200]}")
        return False

def main():
    print("="*60)
    print("VivBot Embedding Models Downloader")
    print("="*60)
    print(f"\nThis will download {len(LOCAL_MODELS)} embedding models.")
    print("Total size: ~5-8 GB")
    print("This may take 10-30 minutes depending on your internet speed.")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
    
    successful = []
    failed = []
    
    for i, model_name in enumerate(LOCAL_MODELS, 1):
        print(f"\n\n[{i}/{len(LOCAL_MODELS)}] Processing model...")
        
        if download_model(model_name):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n✅ Successfully downloaded: {len(successful)}/{len(LOCAL_MODELS)} models")
    
    if successful:
        print("\nWorking models:")
        for model in successful:
            print(f"  ✅ {model}")
    
    if failed:
        print(f"\n❌ Failed to download: {len(failed)} models")
        print("\nFailed models:")
        for model in failed:
            print(f"  ❌ {model}")
        print("\nYou can still use VivBot with the working models.")
        print("The failed models will not appear in the dropdown.")
    
    print("\n" + "="*60)
    
    if len(successful) == 0:
        print("❌ ERROR: No models downloaded successfully!")
        print("Check your internet connection and try again.")
        sys.exit(1)
    else:
        print("✅ Setup complete! You can now start the backend.")
        print("   Run: uvicorn app.main:app --reload")

if __name__ == "__main__":
    main()
