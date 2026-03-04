"""
SmartClaims — Agent Orchestration Service
==========================================
Manages Foundry agent lifecycle, file uploads, and tool routing.
Used by the FastAPI backend.
"""
 
import os
import json
import csv
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    FileSearchTool,
    CodeInterpreterTool,
    CodeInterpreterToolAuto,
    FunctionTool,
)
 
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
 
 
class AgentService:
    """Manages SmartClaims agent and resources."""
 
    def __init__(self):
        self.endpoint = os.environ["PROJECT_ENDPOINT"]
        self.model = os.environ["MODEL_DEPLOYMENT_NAME"]
        self.project_client = AIProjectClient(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
        )
        self.openai_client = self.project_client.get_openai_client()
 
        # State
        self.vector_store_id = None
        self.csv_file_id = None
        self.agent = None
        self.conversation_id = None
        self.claims_data = []  # In-memory for function tools
        self.uploaded_files_info = []
 
    # ─── File Upload ──────────────────────────────────
    def upload_files(self, file_paths: list[dict]) -> dict:
        """
        Upload files for the agent.
 
        Args:
            file_paths: List of {"path": str, "filename": str, "type": "csv"|"doc"}
 
        Each CSV is uploaded for Code Interpreter.
        Each doc (.md, .txt) is indexed in a Vector Store.
        """
        csv_ids = []
        doc_count = 0
        results = []
 
        # Create or reuse Vector Store
        if not self.vector_store_id:
            vs = self.openai_client.vector_stores.create(
                name="SmartClaims-UserDocs"
            )
            self.vector_store_id = vs.id
 
        for item in file_paths:
            fpath = item["path"]
            fname = item["filename"]
            ftype = item["type"]
 
            if ftype == "csv":
                # Upload for Code Interpreter
                f = self.openai_client.files.create(
                    purpose="assistants",
                    file=open(fpath, "rb"),
                )
                csv_ids.append(f.id)
                self.csv_file_id = f.id
 
                # Load into memory for function tools
                with open(fpath, "r", encoding="utf-8") as csvf:
                    self.claims_data = list(csv.DictReader(csvf))
 
                results.append({"file": fname, "type": "csv", "id": f.id,
                                "records": len(self.claims_data)})
 
            elif ftype == "doc":
                # Upload to Vector Store for File Search
                f = self.openai_client.vector_stores.files.upload_and_poll(
                    vector_store_id=self.vector_store_id,
                    file=open(fpath, "rb"),
                )
                doc_count += 1
                results.append({"file": fname, "type": "doc", "id": f.id})
 
        # Create/recreate agent with new resources
        self._create_agent(csv_ids)
 
        # Start fresh conversation
        conv = self.openai_client.conversations.create()
        self.conversation_id = conv.id
 
        self.uploaded_files_info = results
        return {
            "status": "ok",
            "files": results,
            "conversation_id": self.conversation_id,
        }
 
    # ─── Agent Creation ───────────────────────────────
    def _create_agent(self, csv_file_ids: list[str] = None):
        """Create the unified agent with all available tools."""
 
        tools = []
 
        # Code Interpreter (if CSV uploaded)
        if csv_file_ids:
            ci = CodeInterpreterTool(
                container=CodeInterpreterToolAuto(file_ids=csv_file_ids)
            )
            tools.extend(ci.definitions)
 
        # File Search (if docs uploaded)
        if self.vector_store_id:
            fs = FileSearchTool(vector_store_ids=[self.vector_store_id])
            tools.extend(fs.definitions)
 
        # Function Tools (always available)
        funcs = FunctionTool({
            "get_claim_status": self._get_claim_status,
            "calculate_fraud_risk": self._calculate_fraud_risk,
        })
        tools.extend(funcs.definitions)
 
        # Delete old agent if exists
        try:
            self.project_client.agents.delete_agent("smartclaims-webapp")
        except Exception:
            pass
 
        self.agent = self.project_client.agents.create_version(
            agent_name="smartclaims-webapp",
            definition=PromptAgentDefinition(
                model=self.model,
                instructions=(
                    "You are SmartClaims, an AI insurance agent. "
                    "Tools: FILE SEARCH for policy docs, "
                    "CODE INTERPRETER for CSV analytics, "
                    "FUNCTIONS: get_claim_status(claim_id), "
                    "calculate_fraud_risk(incident_type, claim_amount, "
                    "region, days_since_policy_start). "
                    "Always pick the right tool. Be precise."
                ),
                tools=tools,
            ),
        )
 
    # ─── Chat ─────────────────────────────────────────
    def chat(self, message: str) -> str:
        """Send message to agent and return response."""
        if not self.agent:
            return "Please upload files first before chatting."
 
        if not self.conversation_id:
            conv = self.openai_client.conversations.create()
            self.conversation_id = conv.id
 
        try:
            r = self.openai_client.responses.create(
                conversation=self.conversation_id,
                extra_body={"agent_reference": {
                    "name": self.agent.name,
                    "version": self.agent.version,
                    "type": "agent_reference",
                }},
                input=message,
            )
            return r.output_text
        except Exception as e:
            return f"Error: {str(e)}"
 
    # ─── Function Tool Implementations ────────────────
    def _get_claim_status(self, claim_id: str) -> str:
        """Look up claim status from uploaded CSV data."""
        for row in self.claims_data:
            if row.get("claim_id") == claim_id:
                return json.dumps({
                    "claim_id": row.get("claim_id"),
                    "status": row.get("status"),
                    "incident_type": row.get("incident_type"),
                    "claim_amount": row.get("claim_amount"),
                    "approved_amount": row.get("approved_amount"),
                    "fraud_score": row.get("fraud_score"),
                    "region": row.get("region"),
                    "processing_days": row.get("processing_days"),
                }, indent=2)
        return json.dumps({"error": f"Claim '{claim_id}' not found"})
 
    def _calculate_fraud_risk(
        self, incident_type: str, claim_amount: float,
        region: str, days_since_policy_start: int,
    ) -> str:
        """Calculate fraud risk for a new claim."""
        base = {"Auto Collision": 0.15, "Property Damage": 0.10,
                "Medical Claim": 0.12, "Theft": 0.25,
                "Natural Disaster": 0.05, "Liability": 0.18,
                "Fire Damage": 0.20}.get(incident_type, 0.15)
        if claim_amount > 100000: base += 0.15
        elif claim_amount > 50000: base += 0.08
        base += {"North": 0, "South": 0.03, "East": -0.02,
                 "West": 0.05, "Central": 0.01}.get(region, 0)
        if days_since_policy_start < 90: base += 0.10
        score = max(0, min(1, round(base, 2)))
        level = "HIGH" if score >= 0.5 else "MEDIUM" if score >= 0.3 else "LOW"
        return json.dumps({"score": score, "level": level})
 
    # ─── Cleanup ──────────────────────────────────────
    def cleanup(self):
        """Delete all agent resources."""
        try:
            self.project_client.agents.delete_agent("smartclaims-webapp")
        except Exception:
            pass
        if self.vector_store_id:
            try:
                self.openai_client.vector_stores.delete(self.vector_store_id)
            except Exception:
                pass
