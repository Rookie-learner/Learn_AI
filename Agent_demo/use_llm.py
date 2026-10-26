from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.llms import tongyi
from langchain_core.messages import HumanMessage

key = "sk-xxxxxxxxxxxxxxxx"
# llm = tongyi.Tongyi(api_key=key)
#
# answer = llm.invoke("最近一周新疆独库公路北段的天气如何")
# print(answer)

chatLLM = ChatTongyi(
    model="qwen3.7-max",
    api_key=key,
    # dashscope_api_key=key,
)
res = chatLLM.invoke([HumanMessage(content="最近一周新疆独库公路北段的天气如何")])
print(res.content)
