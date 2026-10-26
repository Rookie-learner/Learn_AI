from mcp import StdioServerParameters, stdio_client, ClientSession
import mcp.types as types

server_params = StdioServerParameters(
    command="D:\\Learn\\Learning Project\\learn_AI\\learn_LangChain\\.venv\\Scripts\\python.exe",
    args=["D:\\Learn\\Learning Project\\learn_AI\\learn_LangChain\\Agent_demo\\mcp_server_demo.py"],
    env=None
)

async def handle_sampling_message(message: types.CreateMessageRequestParams) -> types.CreateMessageResult:
    print(f"Received a message: {message}")
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Hello World! from model",
        ),
        model="qwen-plus",
        stopReason="endTurn"
    )

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=handle_sampling_message) as session:
            await session.initialize()

            prompts = await session.list_prompts()
            print(f"prompts: {prompts}")

            tools = await session.list_tools()
            print(f"tools: {tools}")

            resources = await session.list_resources()
            print(f"resources: {resources}")

            result = await session.call_tool("weather", {"city": "西安"})
            print(f"result: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
