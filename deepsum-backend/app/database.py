import json
from typing import List, Optional
from uuid import UUID
import uuid
from supabase import create_client, Client
from datetime import datetime

from config import SUPABASE_URL, SUPABASE_KEY
from models import SummarizationHistoryItem, ChatResponse


class Database:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def save_summarization(self, user_id: UUID, file_name: Optional[str],
                          original_text: str, summary: str, method: str) -> UUID:
        data = {
            "user_id": str(user_id),
            "file_name": file_name,
            "original_text": original_text,
            "summary": summary,
            "created_at": datetime.now().isoformat(),
            "method": method
        }
        
        result = self.supabase.table("summarizations").insert(data).execute()
        return UUID(result.data[0]["id"])

    def get_summarization_history(self, user_id: UUID) -> List[SummarizationHistoryItem]:
        result = self.supabase.table("summarizations") \
            .select("id, file_name, original_text, summary, created_at") \
            .eq("user_id", str(user_id)) \
            .order("created_at", desc=True) \
            .execute()
            
        summaries = []
        for item in result.data:
            summaries.append(SummarizationHistoryItem(
                id=UUID(item["id"]),
                file_name=item.get("file_name"),
                original_text=item["original_text"],
                summary=item["summary"],
                created_at=datetime.fromisoformat(item["created_at"])
            ))
        return summaries

    def delete_summarization(self, summary_id: UUID, user_id: UUID) -> bool:
        result = self.supabase.table("summarizations") \
            .delete() \
            .eq("id", str(summary_id)) \
            .eq("user_id", str(user_id)) \
            .execute()
        return len(result.data) > 0

    def get_summarization(self, summary_id: UUID) -> Optional[dict]:
        result = self.supabase.table("summarizations") \
            .select("*") \
            .eq("id", str(summary_id)) \
            .execute()
        
        if not result.data:
            return None
        return result.data[0]

    def save_chat_message(self, session_id: UUID, user_id: UUID, user_message: str, ai_response: str) -> UUID:
        data = {
            "session_id": str(session_id),
            "user_id": str(user_id),
            "user_message": user_message,
            "ai_response": ai_response,
            "created_at": datetime.now().isoformat()
        }
        
        result = self.supabase.table("chat_messages").insert(data).execute()
        return UUID(result.data[0]["id"])

    def get_chat_history(self, session_id: UUID, user_id: UUID) -> List[ChatResponse]:
        result = self.supabase.table("chat_messages") \
            .select("id, session_id, user_message, ai_response, created_at") \
            .eq("session_id", str(session_id)) \
            .eq("user_id", str(user_id)) \
            .order("created_at", asc=True) \
            .execute()
            
        messages = []
        for item in result.data:
            messages.append(ChatResponse(
                id=UUID(item["id"]),
                session_id=UUID(item["session_id"]),
                user_message=item["user_message"],
                ai_response=item["ai_response"],
                created_at=datetime.fromisoformat(item["created_at"])
            ))
        return messages

    def save_conversational_qna(self, session_id: UUID, summarization_id: UUID, 
                           user_id: UUID, question: str, answer: str, 
                           sequence_number: int) -> UUID:
        data = {
            "session_id": str(session_id),
            "summarization_id": str(summarization_id),
            "user_id": str(user_id),
            "question": question,
            "answer": answer,
            "sequence_number": sequence_number,
            "created_at": datetime.now().isoformat()
        }
        
        result = self.supabase.table("qna_interactions").insert(data).execute()
        return UUID(result.data[0]["id"])

    def get_conversation_history(self, session_id: UUID) -> List[dict]:
        result = self.supabase.table("qna_interactions") \
            .select("question, answer, sequence_number, created_at") \
            .eq("session_id", str(session_id)) \
            .order("sequence_number", desc=False) \
            .execute()
        
        return result.data

    def get_session_count(self, session_id: UUID) -> int:
        result = self.supabase.table("qna_interactions") \
            .select("id", count="exact") \
            .eq("session_id", str(session_id)) \
            .execute()
        
        return result.count or 0

    def get_qna_history(self, summarization_id: UUID) -> List[dict]:
        result = self.supabase.table("qna_interactions") \
            .select("id, question, answer, created_at") \
            .eq("summarization_id", str(summarization_id)) \
            .order("created_at", desc=True) \
            .execute()
        
        return result.data

    def get_conversation_sessions(self, summarization_id: UUID) -> List[dict]:
        result = self.supabase.table("qna_interactions") \
            .select("session_id, question, created_at") \
            .eq("summarization_id", str(summarization_id)) \
            .eq("sequence_number", 1) \
            .order("created_at", desc=True) \
            .execute()
        
        sessions = []
        for item in result.data:
            session_id = item["session_id"]
            count_result = self.supabase.table("qna_interactions") \
                .select("id", count="exact") \
                .eq("session_id", session_id) \
                .execute()
            
            last_result = self.supabase.table("qna_interactions") \
                .select("created_at") \
                .eq("session_id", session_id) \
                .order("sequence_number", desc=True) \
                .limit(1) \
                .execute()
            
            sessions.append({
                "session_id": session_id,
                "first_question": item["question"],
                "last_interaction": last_result.data[0]["created_at"] if last_result.data else item["created_at"],
                "exchange_count": count_result.count or 0
            })
        
        return sessions

    def get_session_details(self, session_id: UUID) -> Optional[dict]:
        result = self.supabase.table("qna_interactions") \
            .select("summarization_id, user_id") \
            .eq("session_id", str(session_id)) \
            .limit(1) \
            .execute()
        
        return result.data[0] if result.data else None

db = Database()

