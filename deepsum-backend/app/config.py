import os
from dotenv import load_dotenv

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "./model/")
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "./model/")

# PDF processing configuration
MAX_PDF_PAGES = 1
MAX_WORDS = 512
