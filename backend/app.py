from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from .rag_service import get_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (ok for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post("/ask")
def ask_question(data: Query):
    try:
        # Pass the query to your RAG service
        answer = get_answer(data.query)
        return {"answer": answer}
    except Exception as e:
        # Check if it's a rate limit error
        if "429" in str(e):
            return {"answer": "The AI is a bit busy (Rate Limit). Please wait a few seconds and try again!"}
        print("RAG ERROR:", e)
        return {"answer": "Sorry, I'm having trouble connecting to my brain right now."}