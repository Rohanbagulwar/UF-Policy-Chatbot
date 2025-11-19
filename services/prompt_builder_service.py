from typing import List, Dict


class PromptBuilder:
    """Builds prompts for OpenAI queries."""
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context documents text.
IMPORTANT RULES:
1. Answer ONLY using information from the provided documents and also if you see a content in an docs so try to create answer from the provided doc content.
2. Do NOT use any external knowledge or information not present in the documents text provided
3. Try to create a Exact and precise answer from the given content you can use your logical reasoning here to frame the answers.
4. Answers should be in 3-4 sentences give me precise answer for the paricular question thik in chain of thoughts to get the answer.
5. Give answers in an normal format do not use any stars or any extra special symbols in an answer also dont give blank lines in the response"""
    
    @staticmethod
    def build_messages(question: str, context: str) -> List[Dict[str, str]]:
        """
        Build message list for OpenAI API.
        
        Args:
            question: User's question
            context: Context text from documents
            
        Returns:
            List of message dictionaries
        """
        user_prompt = f"""Context Documents Content:{context}

Question: {question}

Please answer the question using ONLY the information from the context content above. Do not use any external knowledge."""
        
        return [
            {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]