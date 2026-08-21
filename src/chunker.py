from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

from src.config import (
    PARENT_CHUNK_TOKENS,
    CHILD_CHUNK_TOKENS,
    CHUNK_OVERLAP,
    BASE_DIR
)

class FinancialChunker:
    """
    Implements Parent Document Retrieval (PDR) semantic chunking.
    Splits documents into small child chunks for high-precision vector search, 
    linked to larger parent blocks for context generation.
    """
    def __init__(self, vectorstore, store_path: str = None):
        # 1. Define Parent Splitter (~2,500 tokens / ~10,000 chars) for surrounding narrative context
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_TOKENS * 4,  # Approximately 4 characters per token
            chunk_overlap=200,                   # Overlap safety net between adjacent blocks
            separators=["\n\n", "\n", " ", ""]
        )

        # 2. Define Child Splitter (~500 tokens / ~2,000 chars) with table pipe ('|') preservation
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_TOKENS * 4,
            chunk_overlap=CHUNK_OVERLAP * 4,
            separators=["\n|\n", "|", "\n\n", "\n", " ", ""] # Treats Markdown table boundaries as top priority
        )

        self.vectorstore = vectorstore
        
        # 3. Setup active docstore to hold full parent document context
        self.docstore = InMemoryStore()

    def get_retriever(self) -> ParentDocumentRetriever:
        """
        Constructs and returns the LangChain ParentDocumentRetriever object.
        """
        retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )
        return retriever

    def process_and_add_documents(self, documents: list[Document], retriever: ParentDocumentRetriever):
        """
        Enriches source documents with explicit metadata context headers, 
        ingests them into the PDR retriever, and indexes vectors into Qdrant.
        """
        print("\n" + "="*50)
        print("📊 [CHUNK & STORE PIPELINE] Starting Processing")
        print("="*50)
        
        # Step 1: Prepend context header (e.g., 'Document: MSFT 2024 10K | ') to eliminate orphaned chunks
        enriched_documents = []
        for doc in documents:
            ticker = doc.metadata.get("ticker", "UNKNOWN")
            year = doc.metadata.get("year", "2024")
            doc_type = doc.metadata.get("doc_type", "10K")
            
            context_header = f"Document: {ticker} {year} {doc_type} | "
            
            enriched_doc = Document(
                page_content=context_header + doc.page_content,
                metadata=doc.metadata
            )
            enriched_documents.append(enriched_doc)

        # Step 2: Extract and output source statistics
        total_source_docs = len(enriched_documents)
        total_pages = sum([1 for doc in enriched_documents if "page_number" in doc.metadata])
        
        print(f"📄 Loaded Source Documents : {total_source_docs}")
        print(f"📑 Tracked Document Pages   : {total_pages if total_pages > 0 else 'N/A (Transcript / Stream)'}")
        
        # Step 3: Run PDR ingestion (stores child vectors in Qdrant, parent text in Docstore)
        print("⚙️ Splitting documents into Parent (~2500 tokens) and Child (~500 tokens) chunks...")
        retriever.add_documents(enriched_documents)
        
        print("\n✅ [SUCCESS] Ingestion & Indexing Complete!")
        print(f"📂 Parent Doc Store         : InMemory (Active Session)")
        print(f"🗄️ Vector Database Target   : Qdrant")
        print("="*50 + "\n")