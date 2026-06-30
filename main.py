import requests
import concurrent.futures
import numpy as np

# Local Ollama API endpoints
OLLAMA_API_URL = "http://localhost:11434/api"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

# Split the file into chunks
def load_and_chunk(path: str, chunk_size: int = 1000, stride: int = 800) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = []
    for i in range(0, len(raw_text), stride):
        chunk = raw_text[i : i + chunk_size]
        chunks.append(chunk)
    
    if len(chunks[-1]) < chunk_size:
        chunks.append(raw_text[-chunk_size:])

    print("Successfully loaded and chunked the file.")
    print(f"Number of chunks: {len(chunks)}")
    return chunks

# RETRIEVER PART
def get_local_embedding(text: str) -> list[float]:
    """Hits the local Ollama server to get a text embedding vector."""
    response = requests.post(
        f"{OLLAMA_API_URL}/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status() # Check for errors
    return response.json()["embedding"]

def cosine_similarity(vec_a, vec_b):
    """Calculates how semantically similar two vectors are."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

def embed_all_documents(docs: list[str]) -> list[dict]:
    # Generate embeddings for all documents
    print("Precompute embeddings...")
    embedded_docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_local_embedding, doc): doc for doc in docs}

        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            doc = futures[future]
            vectors = future.result()
            embedded_docs.append({
                "text": doc,
                "embedding": vectors
            })
            if i % 50 == 0: 
                print(f"Embedded {i}/{len(docs)} chunks")

    print("All docs successfully embedded")
    return embedded_docs


def retrieve_best_context(query: str, embedded_docs: list[dict]) -> str:
    """Finds the local document most relevant to the user's query."""
    query_embedding = get_local_embedding(query)
    
    best_score = -1
    best_doc = ""
    
    for doc in embedded_docs:
        doc_embedding = doc["embedding"]
        similarity = cosine_similarity(query_embedding, doc_embedding)
        
        if similarity > best_score:
            best_score = similarity
            best_doc = doc["text"]
            
    print(f"-> Local Retrieval (Similarity Score: {best_score:.4f}):\n '{best_doc}'\n")
    return best_doc

# GENERATOR PART
def ask_local_rag(query: str, embedded_docs: list[dict]):
    """The complete local RAG pipeline."""
    print(f"User Query: {query}")
    
    # 1. Fetch relevant background information locally
    context = retrieve_best_context(query, embedded_docs)
    
    # 2. Craft the augmented prompt
    augmented_prompt = f"""
    You are a helpful assistant. Using the provided context, give a concise but informative answer to the user's question. 
    Synthesize the details mentioned in the text. If the context does not contain enough information to answer the question at all, say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """
    
    # 3. Stream or generate the response from the local Llama model
    response = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={
            "model": LLM_MODEL, 
            "prompt": augmented_prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    
    ai_response = response.json()["response"]
    print(f"Local Llama Response:\n{ai_response}\n")

if __name__ == "__main__":
    file_path = "data/alice_in_wonderland.md"
    
    # Example usage of the RAG pipeline
    raw_chunks = load_and_chunk(file_path)  # Load and chunk the document
    embedded_documents = embed_all_documents(raw_chunks)  # Generate embeddings for document chunks
    ask_local_rag("Who is Alice?", embedded_documents)  # Ask a question using the RAG pipeline
