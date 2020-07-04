from pydantic import BaseModel
from typing import Optional

class TextClassificationRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.5