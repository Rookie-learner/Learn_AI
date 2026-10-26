# -*- coding: utf-8 -*-
"""完整调用本地 MCP 服务：列出并调用 工具 / 资源 / 提示词。

用法:
    python _call_local_mcp.py [url]
默认 url = http://127.0.0.1:8000/mcp
"""
import asyncio
import sys
import traceback

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/mcp"


async def main() -> None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    print(f"连接 {URL}\n")

    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print("[握手] server   =", info.serverInfo.name,
                  "v" + str(info.serverInfo.version))
            print("[握手] protocol =", info.protocolVersion)
            print("[能力]", info.capabilities.model_dump())

            # ---------- 工具 ----------
            tools = await session.list_tools()
            print(f"\n[工具] 共 {len(tools.tools)} 个")
            for t in tools.tools:
                desc = (t.description or "").strip().splitlines()
                props = t.inputSchema.get("properties", {})
                print(f"  · {t.name}  <- {desc[0] if desc else ''}")
                print(f"      参数: {props}")

            print("\n[调用工具]")
            for name, args in [
                ("add", {"a": 3, "b": 4}),
                ("add", {"a": 100, "b": 250}),
                ("weather", {"city": "西安"}),
                ("weather", {"city": "北京"}),
            ]:
                res = await session.call_tool(name, args)
                text = [getattr(c, "text", str(c)) for c in res.content]
                flag = "ERROR" if res.isError else "OK"
                print(f"  [{flag}] {name}({args}) -> {text}")

            # ---------- 资源 ----------
            tpls = await session.list_resource_templates()
            print(f"\n[资源模板] 共 {len(tpls.resourceTemplates)} 个")
            for tpl in tpls.resourceTemplates:
                print(f"  · {tpl.uriTemplate}  ({tpl.name})")

            print("\n[读取资源]")
            for uri in ["greeting://Roy", "greeting://Cline"]:
                try:
                    content = await session.read_resource(uri)
                    vals = [getattr(c, "text", str(c)) for c in content.contents]
                    print(f"  [OK] {uri} -> {vals}")
                except Exception as e:
                    print(f"  [ERROR] {uri} -> {type(e).__name__}: {e}")

            # ---------- 提示词 ----------
            prompts = await session.list_prompts()
            print(f"\n[提示词] 共 {len(prompts.prompts)} 个")
            for p in prompts.prompts:
                print(f"  · {p.name}")

            print("\n>>> 本地 MCP 调用完成 <<<")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()