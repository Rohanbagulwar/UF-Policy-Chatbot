import openai
from typing import List, Dict, Any,Optional


class OpenAIClientManager:
    """Manages OpenAI client operations."""
    
    def __init__(self, config):
        """
        Initialize OpenAI client manager.
        
        Args:
            config: OpenAIConfig object with client settings
        """
        self.config = config
        self.client = None
        
    def initialize(self) -> openai.OpenAI:
        """
        Initialize OpenAI client with custom base URL.
        
        Returns:
            OpenAI client object
        """
        try:
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            
            print('penAI client created successfully')
            print(f'  Base URL: {self.config.base_url}')
            print(f'  Client type: {type(self.client)}')
            
            # Sanity checks
            has_chat = hasattr(self.client, 'chat')
            print(f'  Has chat attribute: {has_chat}')
            
            if has_chat:
                has_completions = hasattr(self.client.chat, 'completions')
                print(f'  Has completions attribute: {has_completions}')
            
            return self.client
        except Exception as e:
            print(f'Error initializing OpenAI client: {e}')
            raise
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create chat completion.
        
        Args:
            messages: List of message dictionaries
            model: Model name (uses config default if None)
            temperature: Temperature (uses config default if None)
            max_tokens: Max tokens (uses config default if None)
            
        Returns:
            Response dictionary
        """
        if self.client is None:
            raise ValueError("Client not initialized. Call initialize() first.")
            
        model = model or self.config.model
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        print(f"\nQuerying {model}...")
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        print("Response received successfully")
        return response