from supabase import create_client, Client
import os
from dotenv import load_dotenv
from fastapi import HTTPException, status

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def get_current_user_id(access_token: str = None) -> str:
    try:
        if access_token:
            # Use the access token to get user info directly
            response = supabase.auth.get_user(access_token)
        else:
            response = supabase.auth.get_user()
        
        if response.user:
            return response.user.id
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No user logged in"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )