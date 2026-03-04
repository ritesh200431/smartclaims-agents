"""
Lab 6: Multi-Tool Agent — Complete SmartClaims
===============================================
Run: python labs/lab6_multi_tool.py
"""
 
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
from utils.config import (
    get_clients, MODEL, CLAIMS_CSV, POLICY_DOC,
    print_header, print_step,
)
from utils.business_functions import get_claim_status, calculate_fraud_risk
from azure.ai.projects.models import (
    PromptAgentDefinition,
    CodeInterpreterTool, CodeInterpreterToolAuto,
    FileSearchTool, FunctionTool,
)
 
 
def main():
    print_header(6, "Multi-Tool Agent — Complete SmartClaims")
    project_client, openai_client = get_clients()
 
    # ═══ PREPARE RESOURCES ══════════════════════════════
    print_step("Step 1: Prepare All Resources")
 
    # 1a: CSV for Code Interpreter
    csv_file = openai_client.files.create(
        purpose="assistants",
        file=open(str(CLAIMS_CSV), "rb"),
    )
    print(f"   ✅ CSV uploaded: {csv_file.id}")
 
    # 1b: Vector Store for File Search
    vs = openai_client.vector_stores.create(name="SmartClaimsPolicies")
    openai_client.vector_stores.files.upload_and_poll(
        vector_store_id=vs.id,
        file=open(str(POLICY_DOC), "rb"),
    )
    print(f"   ✅ Policy indexed in Vector Store: {vs.id}")
 
    # ═══ CONFIGURE TOOLS ════════════════════════════════
    print_step("Step 2: Configure All Tools")
 
    code_interp = CodeInterpreterTool(
        container=CodeInterpreterToolAuto(file_ids=[csv_file.id])
    )
    file_search = FileSearchTool(vector_store_ids=[vs.id])
    funcs = FunctionTool({
        "get_claim_status": get_claim_status,
        "calculate_fraud_risk": calculate_fraud_risk,
    })
 
    all_tools = (
        code_interp.definitions
        + file_search.definitions
        + funcs.definitions
    )
    print(f"   ✅ {len(all_tools)} tool definitions combined")
 
    # ═══ CREATE AGENT ═══════════════════════════════════
    print_step("Step 3: Create Unified SmartClaims Agent")
 
    agent = project_client.agents.create_version(
        agent_name="smartclaims-unified",
        definition=PromptAgentDefinition(
            model=MODEL,
            instructions=(
                "You are SmartClaims — an AI insurance claims agent. "
                "Three tool categories: "
                "(1) FILE SEARCH — policy docs for coverage, exclusions, procedures. "
                "(2) CODE INTERPRETER — claims CSV analysis with Python. "
                "Columns: claim_id, incident_type, claim_amount, approved_amount, "
                "status, fraud_flag, fraud_score, region, policy_type, processing_days. "
                "(3) FUNCTIONS — get_claim_status(claim_id), "
                "calculate_fraud_risk(incident_type, claim_amount, region, "
                "days_since_policy_start). "
                "Pick the right tool(s). You may chain multiple tools."
            ),
            tools=all_tools,
        ),
    )
    print(f"   🚀 Agent: {agent.name} v{agent.version}")
 
    # ═══ TEST SCENARIOS ═════════════════════════════════
    print_step("Step 4: Test Multi-Tool Scenarios")
 
    conv = openai_client.conversations.create()
 
    def ask(q):
        return openai_client.responses.create(
            conversation=conv.id,
            extra_body={"agent_reference": {
                "name": agent.name, "version": agent.version,
                "type": "agent_reference"}},
            input=q,
        )
 
    scenarios = [
        ("Function → Claim Lookup",
         "What is the status of claim CLM-0042?"),
        ("File Search → Policy",
         "Does our policy cover ride-sharing incidents?"),
        ("Code Interpreter → Analytics",
         "How many claims are Under Review? Show status breakdown."),
        ("Function → Fraud Assessment",
         "New Fire Damage claim: $350K, West region, 60-day-old policy. Fraud risk?"),
        ("Multi-Tool → Analysis + Policy",
         "Which incident type has the highest avg claim amount? "
         "What does our policy say about exclusions for that type?"),
    ]
 
    for title, q in scenarios:
        print(f"\n   ┌─ {title}")
        print(f"   │  {q[:80]}")
        r = ask(q)
        out = r.output_text
        if len(out) > 500:
            out = out[:500] + "..."
        print(f"   └─ 🤖 {out}")
 
    # ═══ CLEAN UP ═══════════════════════════════════════
    print_step("Step 5: Clean Up")
    project_client.agents.delete_agent(agent.name)
    openai_client.vector_stores.delete(vs.id)
    print("   ✅ All resources cleaned up")
 
    print(f"\n{'='*65}")
    print("  ✅ Lab 6 Complete!")
    print("  Next → python labs/lab7_tavily_search.py")
    print("    Or → python labs/lab8_production.py")
    print(f"{'='*65}\n")
 
 
if __name__ == "__main__":
    main()
