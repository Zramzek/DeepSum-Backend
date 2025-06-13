# DeepSum Backend API

A FastAPI-based backend for AI-powered PDF summarization with Q&A capabilities using Supabase and Groq.

##BEFORE CLONING AND RUNNING THIS LOCALLY
download the .h5 model from this link :
- https://drive.google.com/drive/folders/1uCTOPmvC3jFUBBBeBhYsDfOvgTBVdxEf?usp=sharing
  
and place it in the root/model/ directory

## Features

- **Authentication**: Supabase Auth with Google/GitHub OAuth
- **PDF Processing**: Upload and process PDFs (1 page, 512 words max)
- **AI Summarization**: Custom .h5 model integration
- **Q&A Chat**: Groq-powered conversations about summaries
- **History Management**: View and manage past summaries and chats

## Project Structure

```
deepsum-backend/
├── main.py                 # FastAPI app entry point
├── config.py              # Configuration and environment variables
├── models.py              # Pydantic models
├── database.py            # Database operations
├── auth.py                # Authentication logic
├── pdf_processor.py       # PDF processing utilities
├── model_service.py       # Custom model integration
├── groq_service.py        # Groq API integration
├── routes/
│   ├── auth.py           # Authentication routes
│   ├── upload.py         # Upload and summarization routes
│   ├── chat.py           # Chat routes
│   └── history.py        # History management routes
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── database_schema.sql   # Database schema
└── README.md            # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

### 3. Supabase Setup

1. Create a new Supabase project
2. Run the SQL in `database_schema.sql` in your Supabase SQL editor
3. Create a storage bucket named "pdfs"
4. Configure OAuth providers (Google, GitHub) in Supabase Auth settings

### 4. Model Setup

Place your `.h5` model and JSON config in the specified paths:

- Update `MODEL_PATH` and `MODEL_CONFIG_PATH` in your `.env`
- Customize `model_service.py` preprocessing and prediction logic

### 5. Run the Application

```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Authentication

- `GET /auth/profile` - Get current user profile

### Upload & Summarization

- `POST /upload/summarize` - Upload PDF and generate summary

### Chat

- `POST /chat/` - Ask questions about a summary
- `GET /chat/history/{session_id}` - Get chat history for a session

### History

- `GET /history/` - Get user's summarization history
- `DELETE /history/{summary_id}` - Delete a summarization

### Utility

- `GET /` - API status
- `GET /health` - Health check

## Authentication

The API uses JWT tokens from Supabase Auth. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## Usage Examples

### Upload and Summarize PDF

```python
import requests

# Upload file
files = {'file': open('document.pdf', 'rb')}
headers = {'Authorization': 'Bearer <your_token>'}

response = requests.post(
    'http://localhost:8000/upload/summarize',
    files=files,
    headers=headers
)

summary_data = response.json()
print(f"Summary: {summary_data['summary']}")
```

### Chat about Summary

```python
import requests

chat_data = {
    "message": "What are the key points?",
    "session_id": "your_summary_id"
}

response = requests.post(
    'http://localhost:8000/chat/',
    json=chat_data,
    headers={'Authorization': 'Bearer <your_token>'}
)

chat_response = response.json()
print(f"AI Response: {chat_response['response']}")
```

## Model Integration

The `model_service.py` file contains placeholder code for your custom .h5 model. You'll need to customize:

1. **Preprocessing**: Update `preprocess_text()` method based on your model's requirements
2. **Prediction**: Modify `generate_summary()` to match your model's input/output format
3. **Configuration**: Use your `model_config.json` for model-specific settings

Example customization:

```python
def preprocess_text(self, text: str):
    # Your model-specific preprocessing
    tokenizer = self.config.get('tokenizer')
    max_length = self.config.get('max_length', 512)

    # Tokenize and pad
    tokens = tokenizer.encode(text)
    padded = pad_sequences([tokens], maxlen=max_length)

    return padded

def generate_summary(self, text: str) -> str:
    processed_input = self.preprocess_text(text)
    prediction = self.model.predict(processed_input)

    # Decode prediction to text
    summary = self.decode_prediction(prediction)
    return summary
```

## Database Schema

The application uses two main tables:

### summarizations

- `id`: UUID (Primary Key)
- `user_id`: UUID (Foreign Key to auth.users)
- `file_name`: Text
- `storage_path`: Text
- `original_text`: Text
- `summary`: Text
- `created_at`: Timestamp

### chat_messages

- `id`: UUID (Primary Key)
- `session_id`: UUID (Foreign Key to summarizations)
- `user_id`: UUID (Foreign Key to auth.users)
- `user_message`: Text
- `ai_response`: Text
- `created_at`: Timestamp

## Security Features

- **Row Level Security (RLS)**: Users can only access their own data
- **JWT Authentication**: Secure token-based authentication
- **File Validation**: PDF format and size restrictions
- **CORS Protection**: Configurable cross-origin requests

## Error Handling

The API returns structured error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:

- `200`: Success
- `400`: Bad Request (validation errors)
- `401`: Unauthorized (authentication required)
- `404`: Not Found
- `500`: Internal Server Error

## Development Notes

### Customization Required

1. **Model Integration**: Adapt `model_service.py` to your specific .h5 model
2. **Preprocessing**: Update text preprocessing based on your model's requirements
3. **CORS Settings**: Configure allowed origins for production
4. **Storage Configuration**: Set up Supabase storage bucket and policies

### Testing

Test your endpoints using the automatic documentation at `http://localhost:8000/docs`

### Production Deployment

1. Set appropriate CORS origins
2. Use environment variables for all secrets
3. Enable HTTPS
4. Set up proper logging
5. Configure rate limiting if needed

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure your .h5 model and config paths are correct
2. **Authentication Failures**: Verify Supabase URL and keys
3. **PDF Processing**: Check file size and format restrictions
4. **Groq API Errors**: Verify API key and model availability

### Logs

Enable detailed logging by setting log level in your environment or adding logging configuration to `main.py`.

## Contributing

1. Follow the existing code structure
2. Add proper error handling
3. Update tests for new features
4. Document any new endpoints
