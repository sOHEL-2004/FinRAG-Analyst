import os
import re
from pathlib import Path
from bs4 import BeautifulSoup
from llama_parse import LlamaParse
from langchain_core.documents import Document
from src.config import LLAMA_CLOUD_API_KEY

class FinancialDocumentLoader:
    def __init__(self):
        # Initialize LlamaParse for complex PDF 10-K financial tables and PDF transcripts
        self.parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=True,
            parsing_instruction="This is a financial document containing text, tables, or speech. Preserve formatting and layout."
        )

    def load_documents(self, data_dir: str | Path = "Data") -> list[Document]:
        """
        Batch loader: Scans the Data directory, extracts ticker/year metadata automatically, 
        and parses all valid financial PDFs and HTML transcripts.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"⚠️ Directory '{data_dir}' not found.")
            return []

        all_docs = []
        supported_files = list(data_path.glob("**/*.pdf")) + list(data_path.glob("**/*.html"))

        print(f"\n📂 [LOADER] Found {len(supported_files)} file(s) in '{data_dir}'. Starting batch process...")

        for file_path in supported_files:
            # Infer ticker (from parent folder name or filename)
            ticker = file_path.parent.name.upper()
            if ticker == "DATA" or len(ticker) > 6:
                match = re.search(r"([a-zA-Z]{2,5})", file_path.stem)
                ticker = match.group(1).upper() if match else "UNKNOWN"

            # Infer year (e.g. 2023, 2024)
            year_match = re.search(r"20\d{2}", str(file_path))
            year = year_match.group(0) if year_match else "2024"

            try:
                docs = self.load_document(file_path, ticker=ticker, year=year)
                all_docs.extend(docs)
            except Exception as e:
                print(f"❌ Failed to load {file_path.name}: {e}")

        return all_docs

    def load_document(self, file_path: str | Path, ticker: str, year: str) -> list[Document]:
        """
        Master router method: Automatically detects file type (.pdf or .html) 
        and document type (10K or TRANSCRIPT) to apply the correct parsing strategy.
        """
        file_path = Path(file_path)
        filename_lower = file_path.name.lower()

        print(f"\n🔍 [LOADER] Processing file: {file_path.name} (Ticker: {ticker}, Year: {year})")

        # Route 1: 10-K PDFs
        if "10k" in filename_lower and file_path.suffix.lower() == ".pdf":
            print(f"📄 Route Identified: SEC 10-K PDF Document")
            return self.load_pdf_10k(file_path, ticker, year)
        
        # Route 2: HTML Transcripts (MSFT, TSLA)
        elif "transcript" in filename_lower and file_path.suffix.lower() == ".html":
            print(f"🎙️ Route Identified: HTML Earnings Call Transcript")
            return self.load_html_transcript(file_path, ticker, year)

        # Route 3: PDF Transcripts (JPM)
        elif "transcript" in filename_lower and file_path.suffix.lower() == ".pdf":
            print(f"🎙️ Route Identified: PDF Earnings Call Transcript")
            return self.load_pdf_transcript(file_path, ticker, year)
        
        else:
            print(f"⚠️ Generic fallback for: {file_path.name}")
            return self.load_pdf_10k(file_path, ticker, year)

    def load_pdf_10k(self, file_path: Path, ticker: str, year: str) -> list[Document]:
        """Parses a 10-K PDF file into structured Markdown documents using LlamaParse."""
        print(f"⏳ Sending 10-K to LlamaParse cloud... (this may take a minute)")
        documents = self.parser.load_data(str(file_path))
        
        langchain_docs = []
        for index, doc in enumerate(documents):
            page_num = doc.metadata.get("page_number", index + 1) if hasattr(doc, "metadata") else index + 1
            langchain_docs.append(
                Document(
                    page_content=doc.text,
                    metadata={
                        "ticker": ticker,
                        "year": year,
                        "doc_type": "10K",
                        "source": file_path.name,
                        "page_number": page_num,
                        "section": "Item 7. MD&A",
                        "chunk_type": "parent",
                    }
                )
            )
        print(f"✅ Successfully parsed 10-K. Total extracted pages/sections: {len(langchain_docs)}")
        return langchain_docs

    def load_html_transcript(self, file_path: Path, ticker: str, year: str) -> list[Document]:
        """Parses HTML transcripts and injects active speaker tags into every sentence."""
        print(f"⏳ Reading and parsing HTML transcript...")
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        current_speaker = "Unknown Speaker"
        speaker_pattern = re.compile(r"^([A-Z][a-zA-Z\s\.\-]+)(\s*[\-\–]\s*[\w\s]+)?:")
        processed_paragraphs = []

        for elem in soup.find_all(["p", "div"]):
            text = elem.get_text(strip=True)
            if not text:
                continue

            match = speaker_pattern.match(text)
            if match:
                current_speaker = match.group(1).strip()

            sentences = re.split(r"(?<=[.!?])\s+", text)
            tagged_sentences = [
                f"[{current_speaker}]: {s}" if not s.startswith(f"[{current_speaker}]") else s
                for s in sentences if s
            ]
            processed_paragraphs.append(" ".join(tagged_sentences))

        full_text = "\n\n".join(processed_paragraphs)
        print(f"✅ Successfully injected speaker tags into HTML transcript.")

        return [
            Document(
                page_content=full_text,
                metadata={
                    "ticker": ticker,
                    "year": year,
                    "doc_type": "TRANSCRIPT",
                    "source": file_path.name,
                    "section": "Earnings Call",
                    "chunk_type": "parent"
                }
            )
        ]

    def load_pdf_transcript(self, file_path: Path, ticker: str, year: str) -> list[Document]:
        """Parses PDF transcripts (like JPM's) using LlamaParse and applies speaker tag injection."""
        print(f"⏳ Sending PDF transcript to LlamaParse cloud...")
        documents = self.parser.load_data(str(file_path))
        raw_text = "\n".join([doc.text for doc in documents])

        current_speaker = "Unknown Speaker"
        speaker_pattern = re.compile(r"^([A-Z][a-zA-Z\s\.\-]+)(\s*[\-\–]\s*[\w\s]+)?:")
        lines = raw_text.split("\n")
        processed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = speaker_pattern.match(line)
            if match:
                current_speaker = match.group(1).strip()

            if not line.startswith(f"[{current_speaker}]"):
                attributed_line = f"[{current_speaker}]: {line}"
            else:
                attributed_line = line

            processed_lines.append(attributed_line)

        full_text = "\n".join(processed_lines)
        print(f"✅ Successfully parsed PDF transcript and injected speaker tags.")

        return [
            Document(
                page_content=full_text,
                metadata={
                    "ticker": ticker,
                    "year": year,
                    "doc_type": "TRANSCRIPT",
                    "source": file_path.name,
                    "section": "Earnings Call",
                    "chunk_type": "parent"
                }
            )
        ]