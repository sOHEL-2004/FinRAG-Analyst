import atexit
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from src.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    BASE_DIR
)

class FinancialVectorStore:
    """
    Manages OpenAI embeddings initialization, local Qdrant database connections,
    collection creation, and safe connection lifecycle management.
    """
    def __init__(self, collection_name: str = "fin_rag_documents"):
        # 1. Initialize OpenAI Embeddings model using config parameters
        print(f"\n🔌 [VECTOR STORE] Initializing embeddings ({EMBEDDING_MODEL})...")
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
        self.collection_name = collection_name
        
        # 2. Define local persistent Qdrant storage folder path
        self.qdrant_path = str(BASE_DIR / "data" / "qdrant_storage")
        print(f"📂 Setting up local Qdrant vector database at: {self.qdrant_path}")
        
        # 3. Connect Qdrant client in local file-storage mode
        self.client = QdrantClient(path=self.qdrant_path)
        
        # 4. Verify collection existence; create new collection if missing (1536 dims for text-embedding-3-small)
        collections = [col.name for col in self.client.get_collections().collections]
        if self.collection_name not in collections:
            print(f"⚙️ Creating new Qdrant collection: '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        else:
            print(f"✅ Found existing Qdrant collection: '{self.collection_name}'")

        # 5. Register automatic exit hook: Closes Qdrant connection before Python unloads system modules (msvcrt)
        atexit.register(self.close)

    def get_vectorstore(self) -> QdrantVectorStore:
        """
        Returns the LangChain QdrantVectorStore instance configured with client and embeddings.
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

    def close(self):
        """
        Safely closes the Qdrant client connection and suppresses 
        Windows module-unloading warnings on exit.
        """
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except (Exception, ModuleNotFoundError, TypeError, AttributeError):
                pass
            finally:
                self.client = None