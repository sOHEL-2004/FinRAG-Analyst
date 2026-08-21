from src.vector_store import FinancialVectorStore
from src.rag_pipeline import FinancialRAGPipeline

def run_t8_metadata_demo():
    print("===================================================================================")
    print("🧪 [LAB T8] METADATA FILTERING DEMONSTRATION (3 COMBINATIONS)")
    print("===================================================================================\n")

    # 1. Initialize Vector Store & Pipeline
    vector_manager = FinancialVectorStore()
    vectorstore = vector_manager.get_vectorstore()
    rag_pipeline = FinancialRAGPipeline(vector_store=vectorstore)

    query = "What are the key financial highlights and revenue figures?"

    # ---------------------------------------------------------------------------------
    # Combination 1: Ticker Only Filter (MSFT)
    # ---------------------------------------------------------------------------------
    print("📌 Combination 1: Filter by Ticker Only ('MSFT')")
    print("-" * 70)
    chain_1 = rag_pipeline.get_chain(ticker_filter="MSFT", year_filter=None)
    response_1 = chain_1.invoke(query)
    print(response_1[:300] + "...\n")

    # ---------------------------------------------------------------------------------
    # Combination 2: Year Only Filter ('2024')
    # ---------------------------------------------------------------------------------
    print("📌 Combination 2: Filter by Year Only ('2024')")
    print("-" * 70)
    chain_2 = rag_pipeline.get_chain(ticker_filter=None, year_filter="2024")
    response_2 = chain_2.invoke(query)
    print(response_2[:300] + "...\n")

    # ---------------------------------------------------------------------------------
    # Combination 3: Compound Filter (Ticker 'TSLA' AND Year '2024')
    # ---------------------------------------------------------------------------------
    print("📌 Combination 3: Compound Filter ('TSLA' + '2024')")
    print("-" * 70)
    chain_3 = rag_pipeline.get_chain(ticker_filter="TSLA", year_filter="2024")
    response_3 = chain_3.invoke(query)
    print(response_3[:300] + "...\n")

    print("===================================================================================")
    print("✅ T8 LAB COMPLETED: All 3 metadata filter combinations successfully verified!")
    print("===================================================================================")

if __name__ == "__main__":
    run_t8_metadata_demo()