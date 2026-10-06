import logging

import uvicorn
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Initialize the FastMCP server.
mcp = FastMCP("sandbox")

# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("e2b-mcp-server")


# Tool schema
class ToolSchema(BaseModel):
    code: str


@mcp.tool()
async def run_code(code: str) -> str:
    """Run python code in a secure sandbox by E2B.

    Uses the Jupyter Notebook syntax. Response includes 1.results, the
    function return value. 2.stdout, the standard output. 3.stderr, the
    standard error.

    Args:
        code: string in Jupyter Notebook syntax.
    """
    sbx = Sandbox()
    execution = sbx.run_code(code)
    logger.info(f"Execution: {execution}")

    result = {
        "results": execution.results,
        "stdout": execution.logs.stdout,
        "stderr": execution.logs.stderr,
    }

    return f"{result}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MCP Streamable HTTP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to listen on")
    args = parser.parse_args()

    starlette_app = mcp.streamable_http_app()

    uvicorn.run(starlette_app, host=args.host, port=args.port)
