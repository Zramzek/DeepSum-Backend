from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from auth import get_current_user
from models import User

from routes import auth, summarize, history, qna

app = FastAPI(
    title="DeepSum API",
    description="API for AI-powered PDF summarization with Q&A capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(summarize.router)
app.include_router(history.router)
app.include_router(qna.router)


@app.get("/")
async def root():
    """API status endpoint"""
    return {"status": "online", "message": "DeepSum API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


import os
os.makedirs("routes", exist_ok=True)

with open("routes/__init__.py", "w") as f:
    f.write("# Routes package\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)