import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

import numpy as np
import matplotlib.pyplot as plt
import umap
from src.vector_store import FinancialVectorStore

def run_umap_lab():
    print("===================================================================================")
    print("🧪 [LAB T5] EMBEDDING VISUALIZATION (BOTH TICKER & DOCTYPE PLOTS)")
    print("===================================================================================\n")
    print("⏳ Fetching ALL chunk vectors from Qdrant DB...")

    vector_manager = FinancialVectorStore()
    client = vector_manager.client
    collection_name = vector_manager.collection_name

    # 1. Fetch ALL 6,600+ vectors
    all_records = []
    next_offset = None

    while True:
        records_batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=500,
            offset=next_offset,
            with_vectors=True,
            with_payload=True
        )
        all_records.extend(records_batch)
        if next_offset is None:
            break

    total_chunks = len(all_records)
    print(f"📊 Total vectors retrieved: {total_chunks}")

    if total_chunks == 0:
        print("⚠️ No vectors found in Qdrant DB. Run ingest_documents.py first!")
        return

    # 2. Extract vectors, tickers, and doc_types
    vectors = np.array([record.vector for record in all_records])
    tickers, doc_types = [], []

    for record in all_records:
        payload = record.payload or {}
        meta = payload.get("metadata", payload)
        if isinstance(meta, dict):
            tickers.append(meta.get("ticker", "UNKNOWN"))
            doc_types.append(meta.get("doc_type", "UNKNOWN"))
        else:
            tickers.append("UNKNOWN")
            doc_types.append("UNKNOWN")

    # 3. Single UMAP Reduction (calculates once for efficiency)
    print("⏳ Running UMAP reduction on full dataset... (10-15 seconds)")
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        metric='cosine', 
        init='random',
        random_state=42
    )
    embedding_2d = reducer.fit_transform(vectors)

    # -----------------------------------------------------------------
    # PLOT 1: BY TICKER (MSFT, JPM, TSLA)
    # -----------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    unique_tickers = list(set(tickers))
    cmap_ticker = plt.colormaps["tab10"]

    for idx, ticker in enumerate(unique_tickers):
        mask = [t == ticker for t in tickers]
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            label=f"{ticker} ({sum(mask)} chunks)",
            color=cmap_ticker(idx % 10),
            alpha=0.6,
            s=30
        )

    plt.title(f"T5 Lab: UMAP Projection by Ticker (All {total_chunks} Chunks)", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Tickers & Counts")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    ticker_png = "T5_UMAP_Ticker_Visualization.png"
    plt.savefig(ticker_png, dpi=300)
    print(f"🖼️ Saved Plot 1: {ticker_png}")
    plt.close()

    # -----------------------------------------------------------------
    # PLOT 2: BY DOCUMENT TYPE (10K vs TRANSCRIPT)
    # -----------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    unique_types = list(set(doc_types))
    cmap_doctype = plt.colormaps["Set1"]

    for idx, d_type in enumerate(unique_types):
        mask = [t == d_type for t in doc_types]
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            label=f"{d_type} ({sum(mask)} chunks)",
            color=cmap_doctype(idx % 10),
            alpha=0.6,
            s=30
        )

    plt.title(f"T5 Lab: UMAP Projection by Document Type (All {total_chunks} Chunks)", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Document Types & Counts")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    doctype_png = "T5_UMAP_DocType_Visualization.png"
    plt.savefig(doctype_png, dpi=300)
    print(f"🖼️ Saved Plot 2: {doctype_png}")
    print("\n✅ Success! Both PNG files are now saved in your project root folder.")
    plt.show()

if __name__ == "__main__":
    run_umap_lab()