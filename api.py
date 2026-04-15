from fastapi import FastAPI
from pydantic import BaseModel
from main import agent
from agent.answer import get_agent_answer
from chat.session import get_session_id
import uuid

app = FastAPI()

class MessageRequest(BaseModel):
    message: str
    session_id: str = ""

class MessageResponse(BaseModel):
    answer: str
    session_id: str

@app.get("/")
def health_check():
    return {"status": "ok", "service": "ЭлектроМощь RAG Agent"}

@app.post("/chat")
def chat(request: MessageRequest) -> MessageResponse:
    session_id = request.session_id or str(uuid.uuid4())

    try:
        answer = get_agent_answer(
            agent=agent,
            session_id=session_id,
            message=request.message
        )
    except Exception as e:
        return MessageResponse(
            answer=f"Ошибка: {str(e)}",
            session_id=session_id
        )

    return MessageResponse(answer=answer, session_id=session_id)