from pathlib import Path
import pypdf
from langchain_core.documents import Document

def preview_chunks_and_embeddings():
    print("==========================================")
    print("📊 FAST CHUNK & EMBEDDING ESTIMATOR")
    print("==========================================")
    
    data_dir = Path("Data")
    if not data_dir.exists():
        print("⚠️ Warning: 'Data' folder not found!")
        return

    # 1. Gather all documents
    raw_docs = []
    pdf_files = list(data_dir.glob("**/*.pdf"))
    
    for pdf in pdf_files:
        try:
            reader = pypdf.PdfReader(pdf)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf.name,
                            "page": page_num + 1,
                            "ticker": pdf.parent.name.upper()
                        }
                    )
                    raw_docs.append(doc)
        except Exception as e:
            print(f"Error reading {pdf.name}: {e}")

    print(f"\n📄 Total Loaded Pages across PDFs: {len(raw_docs)}")

    # 2. Estimate Chunks
    chunk_size = 1000
    chunk_overlap = 200
    
    estimated_total_chunks = 0
    for doc in raw_docs:
        text_length = len(doc.page_content)
        if text_length <= chunk_size:
            estimated_total_chunks += 1
        else:
            chunks_for_doc = max(1, (text_length - chunk_overlap) // (chunk_size - chunk_overlap))
            estimated_total_chunks += chunks_for_doc

    print(f"\n📦 Estimated Total Chunks: ~{estimated_total_chunks}")
    print(f"⚡ Estimated Total Embeddings: ~{estimated_total_chunks}")
    print("==========================================")

if __name__ == "__main__":
    preview_chunks_and_embeddings()