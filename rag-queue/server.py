from  fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from .client.rq_client import queue
from .queue.worker import process_query

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.post("/chat")
def chat(query:str = Query(...,description="User query to be processed")):
    job = queue.enqueue(process_query, query)
    return {"job_id": job.id, "status": "Queued"}



@app.get("/chat/status")
def get_job_status(job_id: str = Query(..., description="Job ID to check status")):
    job = queue.fetch_job(job_id)
    if job is None:
        return {"error": "Job not found"}
    return {"job_id": job.id, "status": job.get_status(), "result": job.result}
    