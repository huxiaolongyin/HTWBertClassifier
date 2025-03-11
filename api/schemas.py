from typing import List, Union

from pydantic import BaseModel


class TextRequest(BaseModel):
    text: Union[str, List[str]]


class ClassificationResponse(BaseModel):
    text: str
    label: str
    score: float
