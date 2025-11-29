from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingModel:
    """Manages embedding model operations."""
    
    def __init__(self, config):
        """
        Initialize embedding model.
        
        Args:
            config: EmbeddingConfig object with model settings
        """
        self.config = config
        self.model = None
        
    def load_model(self) -> SentenceTransformer:
        """
        Load the embedding model from HuggingFace.
        
        Returns:
            Loaded SentenceTransformer model
        """
        print(f"Loading embedding model: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name)
        self.config.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded successfully!")
        print(f" Embedding dimension: {self.config.dimension}")
        return self.model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.encode(texts)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.get_sentence_embedding_dimension()