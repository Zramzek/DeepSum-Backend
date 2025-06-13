from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime


class User(BaseModel):
    id: UUID
    email: str


class SummarizationRequest(BaseModel):
    text: Optional[str] = None


class SummarizationResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    original_text: str
    summary: str
    created_at: datetime = Field(default_factory=datetime.now)
    file_url: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: UUID
    message: str


class ChatResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_message: str
    ai_response: str
    created_at: datetime = Field(default_factory=datetime.now)


class ChatHistoryResponse(BaseModel):
    messages: List[ChatResponse]


class SummarizationHistoryItem(BaseModel):
    id: UUID
    file_name: Optional[str] = None
    original_text: str
    summary: str
    created_at: datetime


class SummarizationHistoryResponse(BaseModel):
    summaries: List[SummarizationHistoryItem]


class QnARequest(BaseModel):
    summarization_id: UUID
    question: str


class QnAResponse(BaseModel):
    question: str
    answer: str
    summarization_id: UUID


class QnAHistoryItem(BaseModel):
    id: UUID
    question: str
    answer: str
    created_at: datetime


class QnAHistoryResponse(BaseModel):
    summarization_id: UUID
    interactions: List[QnAHistoryItem]


class ConversationalQnARequest(BaseModel):
    summarization_id: UUID
    question: str
    session_id: Optional[UUID] = None  


class ConversationalQnAResponse(BaseModel):
    session_id: UUID
    question: str
    answer: str
    summarization_id: UUID
    sequence_number: int
    can_continue: bool  


class HistoryItemSummary(BaseModel):
    id: UUID
    file_name: str | None
    created_at: datetime


class HistoryListResponse(BaseModel):
    items: List[HistoryItemSummary]


class ConversationSession(BaseModel):
    session_id: UUID
    first_question: str
    last_interaction: datetime
    exchange_count: int
    can_continue: bool


class ConversationSessionsResponse(BaseModel):
    summarization_id: UUID
    sessions: List[ConversationSession]


class ConversationHistoryResponse(BaseModel):
    session_id: UUID
    summarization_id: UUID
    interactions: List[QnAHistoryItem]
    can_continue: bool