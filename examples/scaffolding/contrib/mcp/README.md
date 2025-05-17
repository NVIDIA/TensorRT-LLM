USAGE

terminal1: start mcp server
pip install uv
uv pip install httpx mcp
uv init --no-workspace
uv run weather.py

terminal2：run mcp_test
export API_KEY="your-api-key"
pip install dotenv, mcp
python3 mcptest.py