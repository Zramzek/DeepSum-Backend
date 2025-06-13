import groq
from typing import List

from config import GROQ_API_KEY


class GroqService:
    def __init__(self):
        self.client = groq.Client(api_key=GROQ_API_KEY)
        self.model = "llama3-70b-8192"

    async def ask_question(self, context: str, question: str) -> str:
        try:
            prompt = f"""Context: {context}

            Question: {question}

            Please answer the question based only on the information provided in the context. If the answer cannot be determined from the context, say so."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling Groq API: {str(e)}")
            return f"Sorry, I couldn't process your question. Error: {str(e)}"

    async def ask_conversational_question(self, context: str, conversation_history: List[dict], current_question: str) -> str:
        conversation_context = "\n\n=== CONVERSATION HISTORY ===\n"
        for exchange in conversation_history:
            conversation_context += f"Q{exchange['sequence_number']}: {exchange['question']}\n"
            conversation_context += f"A{exchange['sequence_number']}: {exchange['answer']}\n\n"
        
        full_context = f"{context}\n{conversation_context}\n=== CURRENT QUESTION ===\n{current_question}"
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions about documents. Use the provided document content, summary, and conversation history to give accurate, contextual answers. Reference previous exchanges when relevant."
            },
            {
                "role": "user",
                "content": full_context
            }
        ]
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                temperature=0.3,
                max_tokens=1000
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")

groq_service = GroqService()