from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User's medical question",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str):

        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty")

        return value


class ChatResponse(BaseModel):
    answer: str
