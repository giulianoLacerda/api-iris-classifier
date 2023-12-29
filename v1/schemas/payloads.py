from typing import Optional
from pydantic import BaseModel


class ClassifyPayload(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class ModelResponse(BaseModel):
    pred_confidence: float
    pred_label: str
    model_version: str


class ErrorDescription(BaseModel):
    raised: str
    raisedOn: str
    message: str
    code: str

    def __init__(self, raised, raisedOn, message, code):
        super().__init__(raised=raised, raisedOn=raisedOn, message=message, code=code)


class Response(BaseModel):
    message: str
    data: Optional[ModelResponse] = None
    error: Optional[ErrorDescription] = None
    version: str

    def __init__(self, message, data, error, version):
        super().__init__(message=message, data=data, error=error, version=version)
