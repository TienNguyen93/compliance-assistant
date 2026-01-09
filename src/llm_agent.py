# src/llm_agent.py - Create this
"""
LLM integration for answer generation
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.retriever import ComplianceRetriever
import ollama  

class ComplianceAgent:
    """Generate answers using retrieved context"""
    
    def __init__(self, model="llama3.1"):
        self.retriever = ComplianceRetriever()
        self.model = model
        self.client = ollama.Client()
    
    def generate_answer(self, query: str, top_k: int = 5) -> dict:
        """Generate answer with citations"""
        
        # Retrieve relevant context
        results = self.retriever.search(query, top_k=top_k)
        
        # Format context for LLM
        context = self.retriever.get_context_for_llm(query, top_k=top_k)
        
        # Create prompt
        prompt = f"""You are an expert FDA compliance assistant. Answer the question based ONLY on the provided context from FDA guidance documents.

Context from FDA documents:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite specific sources when making claims
- If the context doesn't contain enough information, say so
- Be concise but thorough

Answer:"""
        
        # Generate response
        response = self.client.generate(
            model=self.model,
            prompt=prompt
        )
        
        return {
            'question': query,
            'answer': response['response'],
            'sources': results['results'],
            'num_sources': len(results['results'])
        }


def main():
    """Test the agent"""
    print("=" * 80)
    print("TESTING COMPLIANCE AGENT WITH OLLAMA")
    print("=" * 80 + "\n")
    
    # Initialize agent
    agent = ComplianceAgent(model="llama3.1")
    
    # Test queries
    test_queries = [
        "What does terminal sterilization usually involve?",
        "Which guidance is under section 503B of the Federal Food, Drug, and Cosmetic Act?",
        "What should manufacturers of reusable devices consider?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST QUERY {i}")
        print('='*80)
        print(f"Q: {query}\n")
        
        # Generate answer
        result = agent.generate_answer(query, top_k=3)
        
        print("ANSWER:")
        print("-" * 80)
        print(result['answer'])
        print()
        
        print(f"SOURCES USED ({result['num_sources']}):")
        print("-" * 80)
        for j, source in enumerate(result['sources'], 1):
            print(f"{j}. {source['metadata']['filename']} (Relevance: {source['relevance_score']:.1%})")
        
        print("\n")
        
        # Pause between queries to not overwhelm
        if i < len(test_queries):
            input("Press Enter to continue to next query...")


if __name__ == "__main__":
    main()