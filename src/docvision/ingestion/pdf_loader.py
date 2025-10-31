"""PDF document loading."""

from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader


class PDFLoader:
    """Handle PDF file loading and text extraction."""

    def load_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num} ---\n{page_text}"
        return text

    def load_directory(self, directory: str) -> List[Dict[str, str]]:
        """Load all PDFs from a directory."""
        documents = []
        pdf_dir = Path(directory)

        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                text = self.load_pdf(str(pdf_file))
                documents.append({"content": text, "source": pdf_file.name})
                print(f"✓ Loaded: {pdf_file.name}")
            except Exception as e:
                print(f"✗ Failed: {pdf_file.name} - {e}")

        return documents

# Uncomment for quick testing
# if __name__ == "__main__":
#     loader = PDFLoader()
#     doc = loader.load_pdf("./guide.pdf")
#     print(f"Content Preview: {doc[:100]}...")