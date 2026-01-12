# src/llm_agent.py - Create this
"""
LLM integration for answer generation
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


import config
from src.retriever import ComplianceRetriever
import ollama  
from typing import Dict, List

class ComplianceAgent:
    """Generate answers using retrieved context and Ollama"""
    
    def __init__(self, model: str = "llama3.1"):
        """
        Initialize the agent
        
        Args:
            model: Ollama model name (default: llama3.1)
        """
        self.model = model
        self.retriever = ComplianceRetriever()
        
        # Verify Ollama is available
        try:
            ollama.list()
            print(f"✓ Ollama connected, using model: {model}")
        except Exception as e:
            print(f"✗ Ollama error: {e}")
            print("Make sure Ollama is installed and running!")
            raise
    
    def generate_answer(self, query: str, top_k: int = 5, stream: bool = False) -> Dict:
        """
        Generate an answer to the query using retrieved context
        
        Args:
            query: User's question
            top_k: Number of document chunks to retrieve
            stream: Whether to stream the response (for UI updates)
            
        Returns:
            Dict with question, answer, sources, and metadata
        """
        # Step 1: Retrieve relevant context
        print(f"🔍 Retrieving top {top_k} relevant documents...")
        results = self.retriever.search(query, top_k=top_k)
        
        # Step 2: Format context for LLM
        context = self._format_context(results['results'])
        
        # Step 3: Create system prompt
        system_prompt = """You are an expert FDA compliance assistant specializing in pharmaceutical manufacturing regulations. 

Your role is to:
1. Answer questions accurately based ONLY on the provided FDA guidance documents
2. Cite specific sources when making claims (use [Source N] notation)
3. Be clear when information is not available in the provided context
4. Provide practical, actionable guidance
5. Use professional but accessible language

If the context doesn't contain enough information to fully answer the question, acknowledge this and explain what information is missing."""

        # Step 4: Create user prompt with context
        user_prompt = f"""Context from FDA Guidance Documents:

{context}

---

Question: {query}

Please provide a comprehensive answer based on the context above. Remember to:
- Only use information from the provided sources
- Cite sources using [Source N] where N is the source number
- Be specific and detailed
- If the context is insufficient, clearly state what's missing

Answer:"""

        # Step 5: Generate response
        print("🤖 Generating answer with Ollama...")
        
        if stream:
            # For streaming responses (useful in UI)
            response_stream = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                stream=True
            )
            return {
                'question': query,
                'answer_stream': response_stream,
                'sources': results['results'],
                'num_sources': len(results['results']),
                'context': context
            }
        else:
            # Regular response
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            
            return {
                'question': query,
                'answer': response['message']['content'],
                'sources': results['results'],
                'num_sources': len(results['results']),
                'context': context,
                'model': self.model
            }
    
    def _format_context(self, sources: List[Dict]) -> str:
        """Format retrieved sources into context string"""
        context_parts = []
        
        for i, source in enumerate(sources, 1):
            filename = source['metadata'].get('filename', 'Unknown')
            text = source['text']
            relevance = source['relevance_score']
            
            context_parts.append(
                f"[Source {i}] - {filename} (Relevance: {relevance:.1%})\n{text}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def chat(self, query: str, conversation_history: List[Dict] = None, top_k: int = 5) -> Dict:
        """
        Conversational interface with memory
        
        Args:
            query: Current question
            conversation_history: Previous messages (format: [{'role': 'user/assistant', 'content': '...'}])
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with answer and updated conversation history
        """
        # Retrieve context for current query
        results = self.retriever.search(query, top_k=top_k)
        context = self._format_context(results['results'])
        
        # Build conversation with context
        messages = [
            {'role': 'system', 'content': 'You are an expert FDA compliance assistant.'}
        ]
        
        # Add conversation history if exists
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current query with context
        messages.append({
            'role': 'user',
            'content': f"Context:\n{context}\n\nQuestion: {query}"
        })
        
        # Generate response
        response = ollama.chat(model=self.model, messages=messages)
        
        # Update history
        updated_history = conversation_history.copy() if conversation_history else []
        updated_history.append({'role': 'user', 'content': query})
        updated_history.append({'role': 'assistant', 'content': response['message']['content']})
        
        return {
            'question': query,
            'answer': response['message']['content'],
            'sources': results['results'],
            'conversation_history': updated_history
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