"""
SmartClaims — FastAPI Web Application
======================================
Run locally:  uvicorn app.main:app --reload --port 8000
"""
 
import os
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
 
from app.agent_service import AgentService
 
# ─── App Setup ────────────────────────────────────────
app = FastAPI(title="SmartClaims AI Agent", version="1.0")
templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)
 
# Global agent service instance
agent_svc = AgentService()
 
 
# ─── Models ───────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
 
 
class FraudRequest(BaseModel):
    incident_type: str
    claim_amount: float
    region: str
    days_since_policy_start: int
 
 
class ClaimLookup(BaseModel):
    claim_id: str
 
 
# ─── Routes ───────────────────────────────────────────
 
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the frontend."""
    return templates.TemplateResponse("index.html", {"request": request})
 
 
@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    Upload one or more files for the agent.
 
    Accepted formats:
    - CSV (.csv): Claims data for analytics. Must have columns:
      claim_id, incident_type, claim_amount, status, fraud_flag,
      fraud_score, region, policy_type, processing_days.
      Additional columns are fine.
    - Documents (.md, .txt): Policy/knowledge docs for RAG search.
 
    Multiple files can be uploaded simultaneously.
    """
    tmp_dir = tempfile.mkdtemp()
    file_items = []
 
    try:
        for f in files:
            ext = Path(f.filename).suffix.lower()
            tmp_path = os.path.join(tmp_dir, f.filename)
 
            with open(tmp_path, "wb") as out:
                content = await f.read()
                out.write(content)
 
            if ext == ".csv":
                file_items.append({
                    "path": tmp_path,
                    "filename": f.filename,
                    "type": "csv",
                })
            elif ext in (".md", ".txt"):
                file_items.append({
                    "path": tmp_path,
                    "filename": f.filename,
                    "type": "doc",
                })
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported format: {ext}. "
                             "Use .csv, .md, or .txt"},
                )
 
        if not file_items:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid files uploaded."},
            )
 
        result = agent_svc.upload_files(file_items)
        return result
 
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
 
 
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Send a message to the multi-tool agent."""
    response = agent_svc.chat(req.message)
    return {"response": response}
 
 
@app.post("/api/policy-qa")
async def policy_qa(req: ChatRequest):
    """Ask a question about uploaded policy documents."""
    prompt = (
        f"Using the uploaded policy documents, answer: {req.message}. "
        "Cite specific sections."
    )
    response = agent_svc.chat(prompt)
    return {"response": response}
 
 
@app.post("/api/analytics")
async def analytics(req: ChatRequest):
    """Run analytics on uploaded claims data."""
    prompt = (
        f"Analyze the uploaded CSV data: {req.message}. "
        "Use pandas and matplotlib. Show numbers."
    )
    response = agent_svc.chat(prompt)
    return {"response": response}
 
 
@app.post("/api/claim-lookup")
async def claim_lookup(req: ClaimLookup):
    """Look up a specific claim by ID."""
    response = agent_svc.chat(
        f"Look up claim {req.claim_id}. Show all details."
    )
    return {"response": response}
 
 
@app.post("/api/fraud-risk")
async def fraud_risk(req: FraudRequest):
    """Calculate fraud risk for a new claim."""
    response = agent_svc.chat(
        f"Calculate fraud risk: {req.incident_type} claim for "
        f"${req.claim_amount:,.2f} in {req.region} region, "
        f"policy started {req.days_since_policy_start} days ago."
    )
    return {"response": response}
 
 
@app.on_event("shutdown")
def shutdown():
    """Clean up agent resources on app shutdown."""
    agent_svc.cleanup()
