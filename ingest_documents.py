from src.document_loader import FinancialDocumentLoader
from src.chunker import FinancialChunker
from src.vector_store import FinancialVectorStore

def main():
    print("🚀 [FINRISK AI] Starting Full Pipeline Test...")
    vector_manager = None
    
    try:
        # 1. Initialize Vector Store & Client Connection
        vector_manager = FinancialVectorStore()
        vectorstore = vector_manager.get_vectorstore()

        # 2. Setup Parent-Child Chunker and Retriever
        chunker = FinancialChunker(vectorstore=vectorstore)
        retriever = chunker.get_retriever()

        # 3. Load Documents from 'Data' Directory
        print("\n📥 Loading documents from Data folder...")
        loader = FinancialDocumentLoader()
        documents = loader.load_documents()

        if not documents:
            print("⚠️ No financial documents found in 'Data/' directory! Please add PDFs and try again.")
            return

        # 4. Ingest Documents into Qdrant & Parent DocStore
        chunker.process_and_add_documents(documents, retriever)

        # 5. Run Search Query Test
        query = "What are the primary financial highlights and key revenue figures?"
        print(f"🔍 Testing Retrieval Query: '{query}'")
        
        retrieved_docs = retriever.invoke(query)

        print(f"\n✅ Retrieval successful! Found {len(retrieved_docs)} matching parent context blocks.\n")
        
        if retrieved_docs:
            print("================ Preview Top Parent Chunk Context ================")
            print(retrieved_docs[0].page_content[:400] + "...\n")
            print("==================================================================")

    except Exception as e:
        print(f"❌ Error during execution: {e}")

    finally:
        # Connection safe close karein taaki msvcrt unload warning na aaye
        if vector_manager:
            vector_manager.close()

if __name__ == "__main__":
    main()