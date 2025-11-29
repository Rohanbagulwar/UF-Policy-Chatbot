import chromadb
import os
from typing import Optional, Dict, Any, List


class ChromaDBManager:
    """Manages ChromaDB client and collection operations."""
    
    def __init__(self, config):
        """
        Initialize ChromaDB manager.
        
        Args:
            config: ChromaDBConfig object with database settings
        """
        self.config = config
        self.client = None
        self.collection = None
        
    def initialize(self):
        """Initialize ChromaDB client and collection."""
        # Create persist directory if it doesn't exist
        os.makedirs(self.config.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=self.config.persist_directory)
        print(f"✓ ChromaDB initialized at: {self.config.persist_directory}")
        
        # Get or create collection
        self._get_or_create_collection()
        
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            self.collection = self.client.get_collection(name=self.config.collection_name)
            print(f"✓ Retrieved existing collection: {self.config.collection_name}")
            print(f"  Current documents: {self.collection.count()}")
        except:
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"description": self.config.description}
            )
            print(f"✓ Created new collection: {self.config.collection_name}")
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query ChromaDB collection.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where_filter: Optional filter dictionary
            
        Returns:
            Query results dictionary
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize() first.")
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        return results
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """
        Add documents to collection.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of document IDs
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize() first.")
            
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} documents to collection")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize() first.")
            
        return {
            "name": self.config.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.config.persist_directory
        }