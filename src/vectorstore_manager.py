"""
Vector store management: embedding generation and ChromaDB operations
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


'''
TODO: BM25
'''

# add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config

class VectorStoreManager:
    """Manage embeddings and ChromaDB operations"""
    
    def __init__(self, 
                 collection_name: str = config.COLLECTION_NAME,
                 persist_directory: Path = config.VECTORSTORE_DIR):
        """
        Initialize vector store manager
        
        Args:
            collection_name: Name of ChromaDB collection
            persist_directory: Where to persist the vector database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embedding model
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("✓ Embedding model loaded\n")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Compliance documents for RAG system",
                "hnsw:space": "cosine" 
                }
        )
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def add_documents(self, chunks: List[Dict], batch_size: int = 100):
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of dicts with 'text' and 'metadata' keys
            batch_size: Number of documents to process at once
        """
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Process in batches
        for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
            batch = chunks[i:i + batch_size]
            
            # Extract texts and metadata
            texts = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            
            # Generate IDs (using index for simplicity)
            ids = [f"doc_{i+j}" for j in range(len(batch))]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts, batch_size=32)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"✓ Added {len(chunks)} chunks to collection '{self.collection_name}'")
        print(f"✓ Total documents in collection: {self.collection.count()}\n")
    
    def query(self, 
              query_text: str, 
              n_results: int = config.TOP_K_RESULTS,
              filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Query the vector store
        
        Args:
            query_text: The search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"category": "fda_guidance"})
            
        Returns:
            Dict with 'documents', 'metadatas', 'distances', and 'ids'
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        # Get sample to analyze metadata
        if count > 0:
            sample = self.collection.get(limit=min(count, 100))
            categories = set(meta.get('category', 'unknown') 
                           for meta in sample['metadatas'])
        else:
            categories = set()
        
        return {
            'total_chunks': count,
            'collection_name': self.collection_name,
            'categories': list(categories),
            'embedding_model': config.EMBEDDING_MODEL
        }
    
    def reset_collection(self):
        """Delete and recreate the collection (useful for testing)"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Compliance documents for RAG system"}
        )
        print(f"✓ Reset collection '{self.collection_name}'")


def main():
    """Main function to build the vector store"""
    from src.document_processor import DocumentProcessor
    
    print("=" * 80)
    print("BUILDING VECTOR STORE")
    print("=" * 80 + "\n")
    
    # Step 1: Process documents
    print("Step 1: Processing documents...")
    processor = DocumentProcessor()
    chunks = processor.process_directory(config.RAW_DATA_DIR)
    
    if not chunks:
        print("✗ No documents found! Please add documents to data/raw/")
        return
    
    print(f"✓ Processed {len(chunks)} chunks\n")
    
    # Step 2: Initialize vector store
    print("Step 2: Initializing vector store...")
    vectorstore = VectorStoreManager()
    
    # Optional: Reset if rebuilding
    # vectorstore.reset_collection()
    
    # Step 3: Add documents
    print("Step 3: Generating embeddings and adding to vector store...")
    vectorstore.add_documents(chunks)
    
    # Step 4: Show statistics
    print("Step 4: Collection statistics")
    stats = vectorstore.get_collection_stats()
    print(f"  • Total chunks: {stats['total_chunks']}")
    print(f"  • Categories: {', '.join(stats['categories'])}")
    print(f"  • Embedding model: {stats['embedding_model']}")
    print()
    
    # Step 5: Test query
    print("Step 5: Testing retrieval...")
    test_query = "What are the requirements for cleaning pharmaceutical equipment?"
    print(f"Query: '{test_query}'\n")
    
    results = vectorstore.query(test_query, n_results=3)
    
    print("Top 3 results:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\n--- Result {i} (distance: {distance:.3f}) ---")
        # print(f"Source: {metadata['filename']}")
        # print(f"Category: {metadata['category']}")
        print(f"Text preview: {doc[:200]}...")
    
    print("\n" + "=" * 80)
    print("✓ VECTOR STORE BUILD COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()