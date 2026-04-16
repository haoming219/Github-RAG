from pydantic import BaseModel
from typing import Optional

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class Filters(BaseModel):
    language: Optional[str] = ""
    min_stars: Optional[int] = 0
    topics: Optional[list[str]] = []

class ChatRequest(BaseModel):
    messages: list[Message]
    filters: Optional[Filters] = None

class StarsRange(BaseModel):
    min: int
    max: int

class FilterOptions(BaseModel):
    languages: list[str]
    topics: list[str]
    stars_range: StarsRange
