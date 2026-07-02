"""Request/response schemas. Shapes are non-negotiable per the assignment spec."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class CatalogItem(BaseModel):
    name: str
    url: str
    test_type: str
    description: str = ""
    duration_minutes: Optional[int] = None
    remote_testing: Optional[bool] = None
    adaptive_irt: Optional[bool] = None
    job_levels: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
