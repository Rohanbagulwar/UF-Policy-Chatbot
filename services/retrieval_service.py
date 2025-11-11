from typing import Dict, Any, Optional



class RetrievalService:
    """Main service orchestrating the retrieval pipeline."""
    
    def __init__(
        self,
        embedding_model,
        chroma_manager,
        openai_manager,
        context_builder,
        prompt_builder
    ):
        """
        Initialize retrieval service with all components.
        
        Args:
            embedding_model: EmbeddingModel instance
            chroma_manager: ChromaDBManager instance
            openai_manager: OpenAIClientManager instance
            context_builder: ContextBuilder instance
            prompt_builder: PromptBuilder instance
        """
        self.embedding_model = embedding_model
        self.chroma_manager = chroma_manager
        self.openai_manager = openai_manager
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
    
    def query(
        self,
        question: str,
        n_results: int = 3,
        filter_by_type: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute complete retrieval and generation pipeline.
        
        Args:
            question: User's question
            n_results: Number of documents to retrieve
            filter_by_type: Optional filter by document type
            model: OpenAI model name (optional)
            temperature: Temperature for generation (optional)
            max_tokens: Max tokens for generation (optional)
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # 1. Create query embedding
            query_embedding = self.embedding_model.encode([question])[0].tolist()
            
            # 2. Query ChromaDB
            where_filter = None
            if filter_by_type:
                where_filter = {"type": {"$eq": filter_by_type}}
            
            query_results = self.chroma_manager.query(
                query_embedding=query_embedding,
                n_results=n_results,
                where_filter=where_filter
            )
            
            # 3. Build context
            context = self.context_builder.prepare_context(query_results)
            
            # 4. Build messages
            messages = self.prompt_builder.build_messages(question, context)
            
            # 5. Query OpenAI
            response = self.openai_manager.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 6. Prepare result
            result = {
                'question': question,
                'answer': response.choices[0].message.content,
                'model': model or self.openai_manager.config.model,
                'temperature': temperature or self.openai_manager.config.temperature,
                'tokens_used': {
                    'prompt': response.usage.prompt_tokens,
                    'completion': response.usage.completion_tokens,
                    'total': response.usage.total_tokens
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error in retrieval service: {e}")
            return {
                'question': question,
                'answer': f"Error: {str(e)}",
                'model': model or self.openai_manager.config.model
            }
