from fastapi import FastAPI
from pydantic import BaseModel
from ask_video_clip import ask_question
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str
    video_id: str


@app.post("/ask")
def ask(q: Query):
    answer = ask_question(q.question, q.video_id)
    return {"answer": answer}