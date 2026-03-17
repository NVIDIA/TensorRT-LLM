import logging

import uvicorn
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

# Initialize FastMCP server for Weather tools (SSE)
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
    """Run python code in a secure sandbox by E2B. Using the Jupyter Notebook syntax. Response include 1.results, the function return value. 2.stdout, the standard output. 3.stderr, the standard error.
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


def create_starlette_app(mcp_server: Server,
                         *,
                         debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provided mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse

    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port',
                        type=int,
                        default=8081,
                        help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)
