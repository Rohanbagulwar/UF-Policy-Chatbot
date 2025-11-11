from config import EmbeddingConfig,ChromaDBConfig,OpenAIConfig
from typing import Dict, Any, Optional
from services import retrieval_service,context_builder_service,prompt_builder_service
from models import embedding_model
from vectorstore import chroma_manager
from clients import openai_client

class RetrievalPipeline:
    """Main pipeline class integrating all components."""
    
    def __init__(
        self,
        embedding_config: Optional[EmbeddingConfig] = None,
        chroma_config: Optional[ChromaDBConfig] = None,
        openai_config: Optional[OpenAIConfig] = None
    ):
        """
        Initialize retrieval pipeline with configurations.
        
        Args:
            embedding_config: Configuration for embedding model
            chroma_config: Configuration for ChromaDB
            openai_config: Configuration for OpenAI client
        """
        # Use default configs if not provided
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.chroma_config = chroma_config or ChromaDBConfig()
        self.openai_config = openai_config or OpenAIConfig()
        
        # Initialize components
        self.embedding_model = embedding_model.EmbeddingModel(self.embedding_config)
        self.chroma_manager = chroma_manager.ChromaDBManager(self.chroma_config)
        self.openai_manager = openai_client.OpenAIClientManager(self.openai_config)
        self.context_builder = context_builder_service.ContextBuilder()
        self.prompt_builder = prompt_builder_service.PromptBuilder()
        
        # Initialize retrieval service
        self.service = None
    
    def initialize(self):
        """Initialize all components."""
        print("=" * 60)
        print("INITIALIZING RETRIEVAL PIPELINE")
        print("=" * 60)
        
        # Load embedding model
        self.embedding_model.load_model()
        
        # Initialize ChromaDB
        self.chroma_manager.initialize()
        
        # Initialize OpenAI client
        self.openai_manager.initialize()
        
        # Create retrieval service
        self.service = retrieval_service.RetrievalService(
            embedding_model=self.embedding_model,
            chroma_manager=self.chroma_manager,
            openai_manager=self.openai_manager,
            context_builder=self.context_builder,
            prompt_builder=self.prompt_builder
        )
        
        print("\n✓ Pipeline initialized successfully!")
        print("=" * 60)
    
    def query(
        self,
        question: str,
        n_results: int = 3,
        filter_by_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the pipeline with a question.
        
        Args:
            question: User's question
            n_results: Number of documents to retrieve
            filter_by_type: Optional filter by document type
            
        Returns:
            Dictionary with answer and metadata
        """
        if self.service is None:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        return self.service.query(
            question=question,
            n_results=n_results,
            filter_by_type=filter_by_type
        )


