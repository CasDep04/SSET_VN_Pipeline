import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.tools import tool

# 1. LOAD THE KEY FROM YOUR SPECIFIC FILE
# This tells Python to look into env.txt and treat those lines as system variables
load_dotenv(dotenv_path="env.txt")

# 2. DEFINE A SIMPLE TOOL
@tool
def get_portfolio_summary(client_name: str) -> str:
    """Provides a high-level summary of a client's wealth portfolio."""
    # This is a placeholder for your Wealth Management app logic
    if client_name.lower() == "mario":
        return "Portfolio: $50,000. Allocation: 60% Stocks, 40% Bonds."
    return "Client not found."

# 4. CREATE THE AGENT
agent = create_agent(
    model="claude-haiku-4-5",
    temperature=0.7,
    tools=[get_portfolio_summary],
    system_prompt="You are an AI Financial Advisor. Use your tools to provide accurate data."
)

# 5. TEST IT
if __name__ == "__main__":
    # Ensure the key was actually loaded
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in env.txt")
    else:
        query = "Can you give me a summary of Mario's portfolio?"
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        
        print(f"User: {query}")
        print(f"AI: {response['messages'][-1].content}")