import asyncio
from typing import TypedDict, Annotated
from operator import add

from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain_redis import RedisVectorStore, RedisConfig

import os
from dotenv import load_dotenv
from transformers.cli.serving import response

load_dotenv()
# 创建大语言模型实例
llm = ChatTongyi(
    model="qwen-plus",
    temperature=0.5,
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    streaming=False,
)

NODE_TYPES = ["supervisor", "travel", "couplet", "joke", "other"]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str

def supervisor_node(state: State):
    writer = get_stream_writer()
    writer({"node": ">>> supervisor_node"})
    print("in supervisor_node")
    # 根据用户的问题，对问题进行分类，分类结果保存到type当中
    prompt = """你是一个专业的智能助手，你的任务是根据用户的问题，对问题进行分类，并将任务分给其他Agent执行。
            问题分类结果包括：travel、couplet、joke、other。
            如果用户的问题是和旅游路线规划相关，分类结果为travel。
            如果用户的问题是和对联相关，分类结果为couplet。
            如果用户的问题是和笑话相关，分类结果为joke。
            如果用户的问题是其他，分类结果为other。
            除了这几个选项外，不要返回任何其他的内容。
    """
    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["messages"][0]}
    ]

    # 如果已经有type属性了，表示问题已经交由其他节点处理过了，就可以直接返回
    if "type" in state:
        writer({"supervisor_step": f"已获得分类结果{state['type']}"})
        print(f"已获得分类结果{state['type']}")
        return {"type": END}
    else:
        # 调用大语言模型，对问题进行分类
        response = llm.invoke(prompts)
        typeRes = response.content.strip()
        writer({"supervisor_step": f"进行问题分类：{typeRes}"})
        print(f"supervisor_step 进行问题分类：{typeRes}")
        if typeRes in NODE_TYPES:
            return {"type": typeRes}
        else:
            raise ValueError(f"分类结果{typeRes}不在{NODE_TYPES}中")

async def travel_node(state: State):
    writer = get_stream_writer()
    writer({"node": ">>> travel_node"})
    print("in travel_node")

    system_prompt = """你是一个专业的旅游规划助手，根据用户的问题，生成一个旅游路线规划。请用中文回答，并且限制在100个字以内。"""

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]}
    ]

    # 高德地图MCP服务配置信息
    client = MultiServerMCPClient(
        {
            "amap-maps-streamableHTTP": {
            "url": "https://mcp.amap.com/mcp?key=xxxxxxxxxxxxxx",
            "transport": "streamable_http",
            },

            # 使用标准格式stdio传输，调用MCP服务。实际上就等于执行命令 npx @amap/amap-maps-mcp-server + AMAP_MAPS_API_KEY + 你的问题
            # "amap-maps": {
            #     "command": "npx",
            #     "args": ["-y", "@amap/amap-maps-mcp-server"],
            #     "env": {
            #         "AMAP_MAPS_API_KEY": "xxxxxxxxxxxxxxxxxx"
            #     },
            #     "transport": "stdio",
            # }
        }

    )

    tools = await client.get_tools()
    agent = create_agent(
        model=llm,
        tools=tools,
    )
    response = await agent.ainvoke({"messages": prompts})
    travelRes = response["messages"][-1].content.strip()
    writer({"travel_step": f"规划一个旅游路线：{travelRes}"})
    print(f"travel_step 规划一个旅游路线：{travelRes}")

    return {"messages": [HumanMessage(content=travelRes)],  "type": "travel"}


def couplet_node(state: State):
    writer = get_stream_writer()
    writer({"node": ">>> couplet_node"})
    print("in couplet_node")

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            你是一个专业的联对生成器，你的任务是根据用户给出的上联，设计一个下联。
            回答时，可以参考下面的参考对联。
            参考对联：{samples}
            请用中文回答。
            """),
            ("user", "{text}")
        ]
    )
    query = state["messages"][0]

    embedding_model = DashScopeEmbeddings(model="text-embedding-v1")
    redis_url = "redis://localhost:6379"

    # redis_client = redis.from_url(redis_url)
    # print(redis_client.ping()) # 测试连接 返回True表示连接成功

    config = RedisConfig(
        index_name="couplet",
        redis_url=redis_url
    )

    vector_store = RedisVectorStore(embedding_model, config=config)

    samples = []
    scored_results = vector_store.simulate_search_with_score(query, k=10)
    for doc, score in scored_results:
        # print(f"{doc.page_content} - {score}")
        samples.append(doc.page_content)

    prompt = prompt_template.invoke({"samples": samples, "text": query})
    writer({"couplet_step": f"根据用户问题{query}，设计一个下联：{prompt}"})
    response = llm.invoke(prompt)
    return {"messages": [HumanMessage(content=response.content)],  "type": "couplet"}


def joke_node(state: State):
    writer = get_stream_writer()
    writer({"node": ">>> joke_node"})
    print("in joke_node")

    system_prompt = """ 你是一个笑话大师，根据用户的问题，讲一个不超过50字的笑话。"""

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]}
    ]

    response = llm.invoke(prompts)
    jokeRes = response.content.strip()
    writer({"joke_step": f"讲一个笑话：{jokeRes}"})
    print(f"joke_step 讲一个笑话：{jokeRes}")

    return {"messages": [HumanMessage(content=jokeRes)], "type": "joke"}


def other_node(state: State) -> dict:
    writer = get_stream_writer()
    writer({"node": ">>> other_node"})
    print("in other_node")

    return {"messages": [HumanMessage(content="我暂时无法回答这个问题")], "type": "other"}

# 条件路由函数
def routing_func(state: State):
    writer = get_stream_writer()
    print("in routing_func")
    if state["type"] == "travel":
        writer({"go to": ">>> travel_node"})
        print("go to: travel_node")
        return "travel_node"
    elif state["type"] == "couplet":
        writer({"go to": ">>> couplet_node"})
        print("go to: couplet_node")
        return "couplet_node"
    elif state["type"] == "joke":
        writer({"go to": ">>> joke_node"})
        print("go to: joke_node")
        return "joke_node"
    elif state["type"] == END:
        writer({"go to": ">>> END"})
        print("go to: END")
        return END
    else:
        writer({"go to": ">>> other_node"})
        print("go to: other_node")
        return "other_node"


# 构建图
builder = StateGraph(State)

# 添加节点
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("joke_node", joke_node)
builder.add_node("other_node", other_node)

# 添加边
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", routing_func, ["travel_node", "couplet_node", "joke_node", "other_node", END])
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")

# 构建图
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 执行任务的测试代码

async def main():
    config = {
        "configurable": {"thread_id": "1"}
    }
    async for chunk in graph.astream({"messages":["给我规划一个从西安钟楼到大雁塔的自驾路线"]}, config=config, stream_mode="custom"):
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())

