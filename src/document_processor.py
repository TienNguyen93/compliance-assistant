"""
Document processing: load PDFs/DOCX and chunk them for embedding
"""
import sys
from pathlib import Path
from typing import List, Dict
import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config

class DocumentProcessor:
    """Handle document loading and chunking"""
    
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, 
                 chunk_overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdf(self, file_path: Path) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def load_docx(self, file_path: Path) -> str:
        """Extract text from DOCX"""
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def load_document(self, file_path: Path) -> str:
        """Load document based on file extension"""
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return self.load_pdf(file_path)
        elif suffix == '.docx':
            return self.load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        
        return [
            {
                "text": chunk,
                "metadata": metadata or {}
            }
            for chunk in chunks
        ]
    
    def process_directory(self, directory: Path) -> List[Dict]:
        """Process all documents in a directory"""
        all_chunks = []
        
        for file_path in directory.rglob("*"):
            if file_path.suffix.lower() in ['.pdf', '.docx']:
                print(f"Processing: {file_path.name}")
                
                try:
                    text = self.load_document(file_path)
                    metadata = {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "category": file_path.parent.name
                    }
                    chunks = self.chunk_text(text, metadata)
                    all_chunks.extend(chunks)
                    print(f"  → Created {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {file_path.name}: {e}")
        
        return all_chunks


if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()
    chunks = processor.process_directory(config.RAW_DATA_DIR)
    print(f"\n✓ Total chunks created: {len(chunks)}")