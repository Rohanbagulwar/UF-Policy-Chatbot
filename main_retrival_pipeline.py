import openai
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
import os




class RetrivalPipeline:

    def __init__(self):
        pass

    def load_embedding_model(self,model_name="BAAI/bge-base-en-v1.5"):
        """
        Load BGE embedding model from HuggingFace.
        
        Available BGE models:
        - BAAI/bge-base-en-v1.5 (balanced, 768 dimensions)
        Args:
            model_name: HuggingFace model identifier
        
        Returns:
            SentenceTransformer model
        """
        print(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        print(f"✓ Model loaded successfully!")
        print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model

    def initialize_chromadb(self,persist_directory="./chroma_db", collection_name="policies"):
        """
        Initialize ChromaDB client and create/get collection.
        
        Args:
            persist_directory: Local directory to persist the database
            collection_name: Name of the collection
        
        Returns:
            ChromaDB collection object
        """
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        client = chromadb.PersistentClient(path=persist_directory)
        
        print(f"✓ ChromaDB initialized at: {persist_directory}")
        
        # Get or create collection
        try:
            # Try to get existing collection
            collection = client.get_collection(name=collection_name)
            print(f"✓ Retrieved existing collection: {collection_name}")
            print(f"  Current documents: {collection.count()}")
        except:
            # Create new collection if doesn't exist
            collection = client.create_collection(
                name=collection_name,
                metadata={"description": "UF Policy documents with embeddings"}
            )
            print(f"✓ Created new collection: {collection_name}")
        
        return collection

    def initialize_openai_client(self,api_key: str, base_url: str = 'https://api.ai.it.ufl.edu'):
        """
        Initialize OpenAI client with custom base URL.
        
        Args:
            api_key: Your OpenAI API key
            base_url: Custom base URL for API (default: UF API endpoint)
        
        Returns:
            OpenAI client object
        """
        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            print('✓ OpenAI client created successfully')
            print(f'  Base URL: {base_url}')
            print(f'  Client type: {type(client)}')
            
            # Sanity checks
            has_chat = hasattr(client, 'chat')
            print(f'  Has chat attribute: {has_chat}')
            
            if has_chat:
                has_completions = hasattr(client.chat, 'completions')
                print(f'  Has completions attribute: {has_completions}')
            
            return client
        except Exception as e:
            print(f'Error initializing OpenAI client: {e}')
            return None
        

    def query_chromadb(self,collection, model, query_text, n_results=5, filter_by_type=None):
        """
        Query ChromaDB with a text query.
        
        Args:
            collection: ChromaDB collection
            model: SentenceTransformer model for query embedding
            query_text: Query string
            n_results: Number of results to return
            filter_by_type: Filter results by policy type (optional)
        
        Returns:
            Query results
        """
        # Create query embedding
        query_embedding = model.encode([query_text])[0].tolist()
        
        # Prepare filter
        where_filter = None
        if filter_by_type:
            where_filter = {"type": {"$eq": filter_by_type}}
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        return results

    def prepare_context_from_results(self,query_results: Dict[str, Any]) -> tuple:
        """
        Prepare context and metadata from ChromaDB query results.
        
        Args:
            query_results: Dictionary from ChromaDB query (with ids, documents, metadatas, distances)
        
        Returns:
            Tuple of (context_text, source_documents)
        """
        # Extract documents and metadata
        documents = query_results['documents'][0]
        metadatas = query_results['metadatas'][0]
        distances = query_results['distances'][0]
        
        # Build context text
        context_parts = []
        source_documents = []
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), 1):
            # Add document to context with reference number
            context_parts.append(f"[Document {i}]")
            context_parts.append(f"Title: {metadata['title']}")
            context_parts.append(f"Type: {metadata['type']}")
            context_parts.append(f"Category: {metadata['category']}")
            context_parts.append(f"Content: {doc}")
            context_parts.append("")  # Empty line between documents
            
            # Store source information
            source_documents.append({
                'title': metadata['title'],
                'type': metadata['type'],
                'category': metadata['category'],
                'distance': distance,
                'link': metadata.get('link', 'N/A')
            })
        
        context_text = "\n".join(context_parts)
        
        return context_text

    def query_openai_with_context(self,
        client,
        question: str,
        query_results: Dict[str, Any],
        model: str = "gpt-oss-120b",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Query OpenAI with context from ChromaDB results.
        
        Args:
            client: OpenAI client object
            question: User's question
            query_results: Results from ChromaDB query
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            temperature: Sampling temperature (0-2, lower is more focused)
            max_tokens: Maximum tokens in response
        
        Returns:
            Dictionary with answer and source documents
        """
        try:
            # Prepare context from query results
            context_text = self.prepare_context_from_results(query_results)
            # print(context_text,source_documents)
            
            # Create system prompt
            system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context documents text.

    IMPORTANT RULES:
    1. Answer ONLY using information from the provided documents and also if you see a content in an docs so try to create answer from the provided doc content.
    2. Do NOT use any external knowledge or information not present in the documents text provided
    3. Try to create a answer from the given content its not necessary for exact answers, you can use your logical reasoning here to frame the answers.
    4. Give answers in an normal format do not use any stars or any extra special symbols in an answer also dont give e blank lines in the response

    """
    # 3. If the answer is not in the provided documents, say "I cannot find this information in the provided documents"
            # Create user prompt with context
            user_prompt = f"""Context Documents Content:{context_text}
            Question: {question}
            Please answer the question using ONLY the information from the context content above. Do not use any external knowledge."""

            # Call OpenAI API
            print(f"\n Querying {model}...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Prepare result
            result = {
                'question': question,
                'answer': answer,
                # 'source_documents': source_documents,
                'model': model,
                'temperature': temperature,
                'tokens_used': {
                    'prompt': response.usage.prompt_tokens,
                    'completion': response.usage.completion_tokens,
                    'total': response.usage.total_tokens
                }
            }
            
            print("✓ Response received successfully")
            
            return result
            
        except Exception as e:
            print(f" Error querying OpenAI: {e}")
            return {
                'question': question,
                'answer': f"Error: {str(e)}",
                'source_documents': [],
                'model': model
            }