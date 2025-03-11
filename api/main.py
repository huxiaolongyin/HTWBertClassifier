from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException

from api.model import classifier
from api.schemas import ClassificationResponse, TextRequest

app = FastAPI(
    title="BERT Classification API",
    description="API for text classification using BERT model",
    version="0.0.1",
)


@app.post("/classify", response_model=List[ClassificationResponse])
async def classify_text(request: TextRequest):
    try:
        result = classifier.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Welcome to BERT Classification API. Use /classify endpoint for text classification."
    }


# 如果直接运行此文件
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=4646, reload=True)
