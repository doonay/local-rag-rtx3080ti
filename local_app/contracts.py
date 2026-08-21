from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)
    search_limit: int = Field(default=20, ge=1, le=50)
    temperature: float = Field(default=0.1, ge=0.0, le=1.5)
    max_tokens: int = Field(default=512, ge=32, le=2048)


class Source(BaseModel):
    text: str
    score: float
    document_id: str
    filename: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


SYSTEM_PROMPT = """Ты отвечаешь только на основании предоставленного контекста.
Если контекст не содержит ответа, скажи: «В предоставленных документах нет информации по этому вопросу».
Не добавляй факты из собственных знаний. Отвечай на языке вопроса, кратко и по делу."""
