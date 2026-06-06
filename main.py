from fastapi import FastApi,HTTPException
from pydantic import BaseModel
import concurrent.futures 
from rag_pipeline import load_video, create_or_load_vectorstore, get_answer

app = FastApi(title="OmniScribe AI API", version="1.0.0")

class QueryRequest(BaseModel):
    video_url:str
    question:str
    api_key:str

class QueryResponse(BaseModel):
    answer:str
    video_id:str
    sources_used:int

@app.get("/health")
async def health():
    return {"status":"healthy","service":"OmniScribe AI"}

@app.post("/query",response_model=QueryResponse)
async def query_video(request:QueryRequest):
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(load_video,request.video_url)
            docs,video_id=future.result()
        vectorstore=create_or_load_vectorstore(docs,video_id)
        result,sources = get_answer(vectorstore,request.question,request.api_key)
        return QueryResponse(
            answer=result.answer,
            video_id=video_id,
            sources_used=result.sources_used
        )
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))