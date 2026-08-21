from src.vector_store import FinancialVectorStore
from src.rag_pipeline import FinancialRAGPipeline

def start_chat():
    print("==========================================================")
    print("🤖 FINRISK AI — INTERACTIVE Q&A CHATBOT")
    print("==========================================================")
    print("Type your question below. Type 'exit' or 'quit' to end.\n")

    vector_manager = None
    try:
        # 1. Connect to existing Qdrant Vector Store
        print("⚡ Connecting to Qdrant Storage...")
        vector_manager = FinancialVectorStore()
        vectorstore = vector_manager.get_vectorstore()

        # 2. Initialize RAG Pipeline Chain
        rag_pipeline = FinancialRAGPipeline(vector_store=vectorstore)
        chain = rag_pipeline.get_chain()

        print("✅ Connected! System is ready for questions.\n")

        # 3. Interactive Input Loop
        while True:
            user_query = input("\n💬 Your Question: ").strip()

            if not user_query:
                continue

            if user_query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Exiting Chatbot. Goodbye!")
                break

            print("⏳ Analyzing documents & generating response...")

            try:
                response = chain.invoke(user_query)
                print("\n🤖 [FINRISK AI RESPONSE]:")
                print("-" * 50)
                print(response)
                print("-" * 50)
            except Exception as err:
                print(f"❌ Error generating response: {err}")

    except Exception as e:
        print(f"❌ Connection/Setup Error: {e}")

    finally:
        if vector_manager:
            vector_manager.close()

if __name__ == "__main__":
    start_chat()