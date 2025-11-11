from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from main import RetrievalPipeline

app = FastAPI()


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize pipeline
pipeline = RetrievalPipeline()
pipeline.initialize()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = pipeline.query(
        question=request.question,
        n_results=3
    )
    return {"answer": result['answer']}

# Serve frontend
@app.get("/")
async def serve_frontend():
    return FileResponse('static/index.html')



