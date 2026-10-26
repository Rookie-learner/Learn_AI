from mcp.server.fastmcp import FastMCP

mcp = FastMCP("roymcpdemo")

@mcp.tool()
def add (a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"roy mcp demo called: add({a}, {b})")
    return a + b

@mcp.tool()
def weather(city: str):
    """获取某个城市的天气信息

    Args:
        city (str): 城市名称
    Returns:
        str: 天气信息
       """
    return f"{city}的天气是晴朗的"

@mcp.resource("greeting://{name}")
def greet(name: str) -> str:
    """
    Greet a person by name.
    :param name: 用户姓名
    :return: 欢迎信息
       """
    print(f"roy mcp demo called: greeting({name})")
    return f"你好，{name}！"

if __name__ == "__main__":
    # 以http协议暴露服务
    # mcp.run(transport="streamable-http")
    # 以stdio协议暴露服务
    mcp.run(transport="stdio")