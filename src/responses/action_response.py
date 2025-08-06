from pydantic import BaseModel


class ActionResponse(BaseModel):
    parameters: list[float]
    reasoning: str