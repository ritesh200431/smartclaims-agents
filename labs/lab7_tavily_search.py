"""
Lab 7: Tavily Web Search — Regulatory Intelligence
====================================================
Adds real-time web search to your agent using Tavily API
as a custom Function Tool.
 
Requires: TAVILY_API_KEY in .env
Run: python labs/lab7_tavily_search.py
"""
 
import sys
import os
import json
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
from utils.config import get_clients, MODEL, print_header, print_step
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool
from tavily import TavilyClient
 
 
# ─── Tavily Search Function ──────────────────────────
# This function will be registered as a Function Tool.
# The agent calls it whenever it needs live web data.
tavily_client = None
 
 
def web_search(query: str) -> str:
    """
    Search the web for current information using Tavily API.
 
    Use this tool to find real-time information about insurance
    regulations, industry news, compliance updates, or any
    current events that are not in the uploaded documents.
 
    Args:
        query: The search query (e.g., "latest US insurance
               regulations 2025")
 
    Returns:
        JSON string with search results including titles,
        URLs, and content summaries
    """
    global tavily_client
    if tavily_client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return json.dumps({"error": "TAVILY_API_KEY not set"})
        tavily_client = TavilyClient(api_key=api_key)
 
    try:
        response = tavily_client.search(
            query=query,
            max_results=5,
            include_answer=True,
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:300],
            })
        return json.dumps({
            "answer": response.get("answer", ""),
            "results": results,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
def main():
    print_header(7, "Tavily Web Search — Regulatory Intelligence")
 
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("   ❌ TAVILY_API_KEY not set in .env")
        print("   Get a free key at https://tavily.com")
        return
 
    project_client, openai_client = get_clients()
 
    # ── Step 1: Register web_search as Function Tool ────
    # FunctionTool reads the type hints and docstring from
    # web_search() to auto-generate the JSON schema.
    print_step("Step 1: Register Tavily as Function Tool")
 
    functions = FunctionTool({"web_search": web_search})
    print("   ✅ web_search(query) registered")
 
    # ── Step 2: Create regulatory agent ─────────────────
    print_step("Step 2: Create Regulatory Intelligence Agent")
 
    agent = project_client.agents.create_version(
        agent_name="smartclaims-regulatory",
        definition=PromptAgentDefinition(
            model=MODEL,
            instructions=(
                "You are SmartClaims Regulatory Intelligence Agent. "
                "Use the web_search tool to find current insurance "
                "regulations, compliance requirements, and industry "
                "news. Always cite sources with URLs. Present "
                "information in a clear, actionable format."
            ),
            tools=functions.definitions,
        ),
    )
    print(f"   Agent: {agent.name} v{agent.version}")
 
    # ── Step 3: Query for regulatory updates ────────────
    print_step("Step 3: Query Regulatory Updates")
 
    conv = openai_client.conversations.create()
 
    def ask(q):
        return openai_client.responses.create(
            conversation=conv.id,
            extra_body={"agent_reference": {
                "name": agent.name, "version": agent.version,
                "type": "agent_reference"}},
            input=q,
        )
 
    queries = [
        "What are the latest insurance regulatory changes in the US?",
        "Are there new requirements for AI-powered fraud detection "
        "in insurance?",
        "What are the current NAIC model laws regarding data "
        "privacy in insurance?",
    ]
 
    for i, q in enumerate(queries, 1):
        print(f"\n   ┌─ Query {i}: {q}")
        r = ask(q)
        out = r.output_text
        if len(out) > 500:
            out = out[:500] + "..."
        print(f"   └─ 🌐 {out}")
 
    # ── Step 4: Clean up ────────────────────────────────
    print_step("Step 4: Clean Up")
    project_client.agents.delete_agent(agent.name)
    print("   ✅ Agent deleted")
 
    print(f"\n{'='*65}")
    print("  ✅ Lab 7 Complete!")
    print("  Next → python labs/lab8_production.py")
    print(f"{'='*65}\n")
 
 
if __name__ == "__main__":
    main()
