"""
Retriever module: Clean interface for querying the vector store
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import config
from src.vectorstore_manager import VectorStoreManager
from typing import List, Dict, Optional

class ComplianceRetriever:
    """High-level interface for retrieving compliance information"""
    
    def __init__(self):
        """Initialize the retriever with vector store"""
        self.vectorstore = VectorStoreManager()
        print(f"✓ Retriever initialized with {self.vectorstore.collection.count()} documents")
    
    def search(self, 
               query: str, 
               top_k: int = config.TOP_K_RESULTS,
               category_filter: Optional[str] = None) -> Dict:
        """
        Search for relevant documents
        
        Args:
            query: The search question
            top_k: Number of results to return
            category_filter: Optional filter by category (e.g., 'fda_guidance')
            
        Returns:
            Dict with results and metadata
        """
        # Build metadata filter if category specified
        metadata_filter = {"category": category_filter} if category_filter else None
        
        # Query vector store
        results = self.vectorstore.query(
            query_text=query,
            n_results=top_k,
            filter_metadata=metadata_filter
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return {
            'query': query,
            'num_results': len(formatted_results),
            'results': formatted_results
        }
    
    def get_context_for_llm(self, query: str, top_k: int = 5) -> str:
        """
        Get formatted context string for LLM prompting
        
        Args:
            query: The user's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Formatted context string with sources
        """
        search_results = self.search(query, top_k=top_k)
        
        context_parts = []
        for i, result in enumerate(search_results['results'], 1):
            source = result['metadata'].get('filename', 'Unknown')
            text = result['text']
            score = result['relevance_score']
            
            context_parts.append(
                f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        return self.vectorstore.get_collection_stats()


def main():
    """Test the retriever with FDA-specific queries"""
    print("=" * 80)
    print("TESTING RETRIEVER - FDA GUIDANCE QUERIES")
    print("=" * 80 + "\n")
    
    retriever = ComplianceRetriever()
    
    # FDA-focused test queries
    test_queries = [
        "What are the requirements for cleaning validation in pharmaceutical manufacturing?",
        "What is process validation and why is it important?",
        "What are the requirements for sterile drug product manufacturing?",
        "How should equipment cleaning procedures be documented?",
        "What are the critical process parameters in pharmaceutical manufacturing?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print('='*80)
        
        results = retriever.search(query, top_k=3)
        
        print(f"\nFound {results['num_results']} results:\n")
        
        for j, result in enumerate(results['results'], 1):
            print(f"--- Result {j} ---")
            print(f"Source: {result['metadata']['filename']}")
            print(f"Relevance Score: {result['relevance_score']:.3f}")
            print(f"Text Preview: {result['text'][:200]}...")
            print()
    
    # Show formatted context example
    print("\n" + "="*80)
    print("EXAMPLE: Formatted Context for LLM")
    print("="*80)
    context = retriever.get_context_for_llm(test_queries[0], top_k=2)
    print(context[:500] + "...\n")
    
    # Show stats
    print("="*80)
    print("RETRIEVER STATISTICS")
    print("="*80)
    stats = retriever.get_stats()
    for key, value in stats.items():
        print(f"  • {key}: {value}")


if __name__ == "__main__":
    main()