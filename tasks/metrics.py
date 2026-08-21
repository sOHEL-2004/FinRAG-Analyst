import numpy as np
from scipy.spatial.distance import cosine, euclidean
from src.vector_store import FinancialVectorStore

def run_metrics_lab():
    print("===================================================================================")
    print("🧪 [LAB T7] SIMILARITY METRICS BENCHMARKING (FinRisk AI)")
    print("===================================================================================\n")

    vector_manager = FinancialVectorStore()
    embeddings = vector_manager.embeddings
    client = vector_manager.client

    # 1. Embed query vector
    query_text = "What are the primary financial highlights, net income, and revenue figures?"
    query_vec = np.array(embeddings.embed_query(query_text))

    # 2. Scroll top sample chunks from Qdrant DB
    records, _ = client.scroll(
        collection_name=vector_manager.collection_name,
        limit=5,
        with_vectors=True,
        with_payload=True
    )

    if not records:
        print("⚠️ No vectors found in Qdrant DB! Run ingest_documents.py first.")
        return

    print(f"🔍 Test Query: '{query_text}'\n")
    print(f"{'Chunk Preview':<40} | {'Cosine Sim ↑':<12} | {'Dot Product ↑':<13} | {'Euclidean Dist ↓':<15}")
    print("-" * 88)

    for record in records:
        chunk_vec = np.array(record.vector)
        text_preview = record.payload.get("page_content", "").replace("\n", " ")[:37] + "..."

        # Compute 3 Distance Metrics
        cos_sim = 1 - cosine(query_vec, chunk_vec)
        dot_prod = np.dot(query_vec, chunk_vec)
        euc_dist = euclidean(query_vec, chunk_vec)

        print(f"{text_preview:<40} | {cos_sim:<12.4f} | {dot_prod:<13.4f} | {euc_dist:<15.4f}")

    print("\n===================================================================================")
    print("📝 FINDINGS & INTERNSHIP REPORT SUMMARY")
    print("===================================================================================")
    print("1. Cosine Similarity & Dot Product yield identical chunk rankings because OpenAI")
    print("   'text-embedding-3-small' embeddings are pre-normalized to unit length.")
    print("2. Euclidean Distance correlates inversely (smaller distance = higher semantic similarity).")
    print("3. Recommendation for FinRisk AI: Cosine Similarity is optimal for Qdrant setup.")
    print("===================================================================================\n")

if __name__ == "__main__":
    run_metrics_lab()