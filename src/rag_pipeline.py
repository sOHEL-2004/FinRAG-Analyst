from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client.http import models

# Import from our dedicated modular files
from src.llm_engine import get_llm_engine
from src.prompt_templates import get_financial_prompt_template

class FinancialRAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
        # Pull the clean LLM engine (temperature 0.0)
        self.llm = get_llm_engine(temperature=0.0)
        
        # Pull the professional financial prompt template
        self.prompt = get_financial_prompt_template()

    def get_chain(self, ticker_filter: str = None, year_filter: str = None):
        """
        Builds a modern LCEL RAG chain with Qdrant metadata pre-filtering (Task 8).
        """
        # Construct Qdrant native filter conditions if filters are applied
        qdrant_filter = None
        conditions = []
        
        if ticker_filter:
            conditions.append(models.FieldCondition(key="metadata.ticker", match=models.MatchValue(value=ticker_filter)))
        if year_filter:
            conditions.append(models.FieldCondition(key="metadata.year", match=models.MatchValue(value=year_filter)))
            
        if conditions:
            qdrant_filter = models.Filter(must=conditions)

        # Configure retriever with k=4 chunks and metadata filters
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": qdrant_filter
            }
        )

        # 🔧 FIX HERE: Prepend metadata header to every text chunk!
        def format_docs(docs):
            formatted_chunks = []
            for doc in docs:
                ticker = doc.metadata.get("ticker", "UNKNOWN")
                year = doc.metadata.get("year", "UNKNOWN")
                doc_type = doc.metadata.get("doc_type", "UNKNOWN")
                source = doc.metadata.get("source", "UNKNOWN")
                page_num = doc.metadata.get("page_number", "")
                
                # Metadata header string
                header = f"[DOCUMENT: {ticker} | YEAR: {year} | TYPE: {doc_type} | FILE: {source}]"
                if page_num:
                    header += f" [PAGE: {page_num}]"
                
                # Merge header with page content
                chunk_text = f"{header}\n{doc.page_content}"
                formatted_chunks.append(chunk_text)

            return "\n\n---\n\n".join(formatted_chunks)

        # Build modern LCEL execution chain using the pipe operator
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain