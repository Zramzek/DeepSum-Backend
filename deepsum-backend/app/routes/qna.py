from fastapi import APIRouter, HTTPException, status
from uuid import UUID, uuid4
from datetime import datetime

from models import (

    QnAHistoryItem,
    QnAHistoryResponse,
    ConversationalQnARequest,
    ConversationalQnAResponse,
    ConversationSession, 
    ConversationSessionsResponse, 
    ConversationHistoryResponse
)
from database import db
from groq_service import groq_service

router = APIRouter(prefix="/qna", tags=["qna"])

@router.post("/", response_model=ConversationalQnAResponse)
async def conversational_qna(request: ConversationalQnARequest):
    try:
        MAX_EXCHANGES = 10 
        
        summarization = db.get_summarization(request.summarization_id)
        if not summarization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Summarization not found"
            )
        
        if request.session_id:
            session_id = request.session_id
            exchange_count = db.get_session_count(session_id)
            if exchange_count >= MAX_EXCHANGES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Maximum {MAX_EXCHANGES} exchanges per session reached"
                )
            sequence_number = exchange_count + 1
        else:
            session_id = uuid4()
            sequence_number = 1
        
        conversation_history = db.get_conversation_history(session_id) if request.session_id else []
        
        document_context = f"Original Text: {summarization['original_text']}\n\nSummary: {summarization['summary']}"
        
        answer = await groq_service.ask_conversational_question(
            context=document_context,
            conversation_history=conversation_history,
            current_question=request.question
        )
        
        user_id = UUID(summarization['user_id'])
        interaction_id = db.save_conversational_qna(
            session_id=session_id,
            summarization_id=request.summarization_id,
            user_id=user_id,
            question=request.question,
            answer=answer,
            sequence_number=sequence_number
        )
        
        can_continue = sequence_number < MAX_EXCHANGES
        
        return ConversationalQnAResponse(
            session_id=session_id,
            question=request.question,
            answer=answer,
            summarization_id=request.summarization_id,
            sequence_number=sequence_number,
            can_continue=can_continue
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing conversational question: {str(e)}"
        )

@router.get("/history/{summarization_id}", response_model=QnAHistoryResponse)
async def get_qna_history_for_summary(summarization_id: UUID,):
    try:
        summarization = db.get_summarization(summarization_id)
        if not summarization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Summarization not found"
            )
        
        history_data = db.get_qna_history(summarization_id)
        
        interactions = [
            QnAHistoryItem(
                id=UUID(item["id"]),
                question=item["question"],
                answer=item["answer"],
                created_at=datetime.fromisoformat(item["created_at"])
            )
            for item in history_data
        ]
        
        return QnAHistoryResponse(
            summarization_id=summarization_id,
            interactions=interactions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving Q&A history: {str(e)}"
        )


@router.get("/sessions/{summarization_id}", response_model=ConversationSessionsResponse)
async def get_conversation_sessions(summarization_id: UUID):
    try:
        summarization = db.get_summarization(summarization_id)
        if not summarization:
            raise HTTPException(status_code=404, detail="Summarization not found")
        
        sessions_data = db.get_conversation_sessions(summarization_id)
        
        sessions = [
            ConversationSession(
                session_id=UUID(session["session_id"]),
                first_question=session["first_question"],
                last_interaction=datetime.fromisoformat(session["last_interaction"]),
                exchange_count=session["exchange_count"],
                can_continue=session["exchange_count"] < 10
            )
            for session in sessions_data
        ]
        
        return ConversationSessionsResponse(
            summarization_id=summarization_id,
            sessions=sessions
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sessions: {str(e)}")


@router.get("/conversation/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_by_session(session_id: UUID):
    try:
        conversation_data = db.get_conversation_history(session_id)
        
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        first_interaction = db.get_session_details(session_id)
        if not first_interaction:
            raise HTTPException(status_code=404, detail="Session not found")
        
        interactions = [
            QnAHistoryItem(
                id=UUID(item.get("id", str(uuid4()))),  
                question=item["question"],
                answer=item["answer"],
                created_at=datetime.fromisoformat(item["created_at"])
            )
            for item in conversation_data
        ]
        
        return ConversationHistoryResponse(
            session_id=session_id,
            summarization_id=UUID(first_interaction["summarization_id"]),
            interactions=interactions,
            can_continue=len(interactions) < 10
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")