from fastapi import APIRouter, HTTPException, Depends
from typing import List
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime

from database import Database
from auth import get_current_user
from models import User, HistoryItemSummary, HistoryListResponse

router = APIRouter(prefix="/history", tags=["history"])

db = Database()

@router.get("/", response_model=HistoryListResponse)
async def get_all_history():
    current_user_id= "10668e40-5071-444f-b27c-e686a8284a95"
    try:
        full_history = db.get_summarization_history(current_user_id)
        
        history_items = [
            HistoryItemSummary(
                id=item.id,
                file_name=item.file_name,
                created_at=item.created_at
            )
            for item in full_history
        ]
        
        return HistoryListResponse(items=history_items)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@router.get("/{item_id}", response_model=HistoryItemSummary)
async def get_history_item(item_id: UUID):
    try:
        summarization = db.get_summarization(item_id)
        
        current_user_id= "10668e40-5071-444f-b27c-e686a8284a95"

        if not summarization:
            raise HTTPException(status_code=404, detail="History item not found")
        
        if summarization.get("user_id") != str(current_user_id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return HistoryItemSummary(
            id=UUID(summarization["id"]),
            file_name=summarization.get("file_name"),
            created_at=datetime.fromisoformat(summarization["created_at"])
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history item: {str(e)}")