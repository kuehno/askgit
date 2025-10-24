import asyncio
import os
import sys
import textwrap

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv

load_dotenv()

MODEL = LiteLLM(id=os.getenv("LITELLM_MODEL_ID"), name="LiteLLM", temperature=None)


async def run_agent(message: str):
    # Initialize the MCP server
    async with (
        MCPTools(command="uv run fastmcp run server.py", env=os.environ) as mcp_tools,
    ):
        agent = Agent(
            model=MODEL,
            tools=[mcp_tools],
            instructions=textwrap.dedent("""
            You are a coding assistant who is specialized in answering the users
            coding requests with additional input from private and public github repos.
            For finding information about code or repositories use the search_similar tool.
            """),
        )

        await agent.aprint_response(
            message,
            markdown=True,
            show_reasoning=False,
            stream=True,
            stream_intermediate_steps=True,
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run agno_agent_demo.py '<your query>'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    asyncio.run(run_agent(query))
