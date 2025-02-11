import faiss
import numpy as np
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient for large-scale indexing

# File paths for saving index
INDEX_FILE = "full_wiki_faiss.index"
METADATA_FILE = "full_wiki_metadata.json"

def load_wikipedia_dump():
    """Load the full Wikipedia dataset from Hugging Face."""
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    return dataset

def build_faiss_index(titles, texts, urls):
    """Convert Wikipedia full text into embeddings and store in FAISS index."""
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=256, num_processes=32)
    d = embeddings.shape[1]  # Dimension of embeddings

    index = faiss.IndexFlatL2(d)  # L2 distance index
    index.add(embeddings)

    return index, titles, texts, urls

def save_faiss_index(index, titles, texts, urls, index_file=INDEX_FILE, metadata_file=METADATA_FILE):
    """Save FAISS index and metadata (titles, texts, urls)."""
    faiss.write_index(index, index_file)

    metadata = [{"title": t, "text": tx, "url": u} for t, tx, u in zip(titles, texts, urls)]
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

def load_faiss_index(index_file=INDEX_FILE, metadata_file=METADATA_FILE):
    """Load FAISS index and metadata."""
    index = faiss.read_index(index_file)

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    titles = [item["title"] for item in metadata]
    texts = [item["text"] for item in metadata]
    urls = [item["url"] for item in metadata]

    return index, titles, texts, urls

def query_faiss_index(index, titles, texts, urls, query_text, k=5):
    """Query FAISS index with text and return top K Wikipedia articles with full text and URLs."""
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        if idx < len(titles):  # Avoid out-of-bounds errors
            results.append({
                "title": titles[idx],
                "text": texts[idx],
                "url": urls[idx]
            })
    return results

if __name__ == "__main__":

    # Step 1: Load Wikipedia dataset
    ds = load_wikipedia_dump()

    # Step 3: Build FAISS index
    faiss_index, wiki_titles, wiki_texts, wiki_urls = build_faiss_index(ds["title"], ds["text"], ds["url"])
    save_faiss_index(faiss_index, wiki_titles, wiki_texts, wiki_urls)

    print("Wikipedia FAISS index built and saved.")

    # Step 4: Load FAISS index
    index, titles, texts, urls = load_faiss_index()
    print("Index loaded.")

    # Step 5: Query Wikipedia
    query = "What is artificial intelligence?"
    results = query_faiss_index(index, titles, texts, urls, query)

    print("\nTop matches:")
    for result in results:
        print(f"\nTitle: {result['title']}\nURL: {result['url']}\nText:\n{result['text'][:500]}...\n")

