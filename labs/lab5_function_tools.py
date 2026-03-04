"""
Lab 5: Function Tool Agent — Custom Business Logic
====================================================
Run: python labs/lab5_function_tools.py
"""
 
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
from utils.config import get_clients, MODEL, print_header, print_step
from utils.business_functions import get_claim_status, calculate_fraud_risk
from azure.ai.projects.models import (
    PromptAgentDefinition, FunctionTool, ToolSet,
)
 
 
def main():
    print_header(5, "Function Tool Agent — Custom Business Logic")
    project_client, openai_client = get_clients()
 
    # ── Step 1: Create FunctionTool ─────────────────────
    # FunctionTool auto-generates JSON schemas from your
    # Python type hints and docstrings. No manual schema.
    print_step("Step 1: Register Functions")
 
    user_functions = {
        "get_claim_status": get_claim_status,
        "calculate_fraud_risk": calculate_fraud_risk,
    }
    functions = FunctionTool(user_functions)
    print("   ✅ get_claim_status(claim_id)")
    print("   ✅ calculate_fraud_risk(incident_type, claim_amount, region, days)")
 
    # ── Step 2: Create ToolSet for auto-execution ───────
    # ToolSet enables the SDK to call your functions
    # automatically when the agent requests them.
    print_step("Step 2: Create ToolSet (Auto-Execution)")
 
    toolset = ToolSet()
    toolset.add(functions)
    print("   ✅ Auto-execution enabled")
 
    # ── Step 3: Create agent ────────────────────────────
    print_step("Step 3: Create Agent")
 
    agent = project_client.agents.create_version(
        agent_name="smartclaims-functions",
        definition=PromptAgentDefinition(
            model=MODEL,
            instructions=(
                "You are SmartClaims Operations Agent. Tools available: "
                "(1) get_claim_status — look up any claim by ID (CLM-XXXX). "
                "(2) calculate_fraud_risk — assess fraud risk for new claims. "
                "Always use tools when relevant. Present results clearly."
            ),
            tools=functions.definitions,
        ),
    )
    print(f"   Agent: {agent.name} v{agent.version}")
 
    # ── Step 4: Test function calls ─────────────────────
    print_step("Step 4: Test Function Calls")
 
    conversation = openai_client.conversations.create()
 
    def ask(q):
        return openai_client.responses.create(
            conversation=conversation.id,
            extra_body={"agent_reference": {
                "name": agent.name, "version": agent.version,
                "type": "agent_reference"}},
            input=q,
        )
 
    tests = [
        ("Claim Lookup", "What is the status of claim CLM-0042?"),
        ("Fraud Risk", "Calculate fraud risk for a $85,000 Theft "
         "claim from the West region. Policy started 45 days ago."),
        ("High-Risk Claim", "New Fire Damage claim: $350,000, "
         "Central region, policy only 20 days old. Fraud risk?"),
        ("Combined", "Look up CLM-0100. Is the fraud score suspicious?"),
    ]
 
    for title, q in tests:
        print(f"\n   ┌─ {title}: {q[:70]}...")
        r = ask(q)
        out = r.output_text
        if len(out) > 500:
            out = out[:500] + "..."
        print(f"   └─ 🤖 {out}")
 
    # ── Step 5: Clean up ────────────────────────────────
    print_step("Step 5: Clean Up")
    project_client.agents.delete_agent(agent.name)
    print("   ✅ Agent deleted")
 
    print(f"\n{'='*65}")
    print("  ✅ Lab 5 Complete!")
    print("  Next → python labs/lab6_multi_tool.py")
    print(f"{'='*65}\n")
 
 
if __name__ == "__main__":
    main()
