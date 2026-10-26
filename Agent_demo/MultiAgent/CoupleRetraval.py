# 对联数据RAG检索

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi


query = "帮我对个对联： 瑞雪兆丰年"

load_dotenv()

embedding_model = DashScopeEmbeddings(model="text-embedding-v1")
redis_url = "redis://localhost:6379"

# redis_client = redis.from_url(redis_url)
# print(redis_client.ping()) # 测试连接 返回True表示连接成功

from langchain_redis import RedisVectorStore, RedisConfig
config = RedisConfig(
    index_name = "couplet",
    redis_url = redis_url
)

vector_store = RedisVectorStore(embedding_model, config=config)
samples = []

scored_results = vector_store.simulate_search_with_score(query, k=10)
for doc, score in scored_results:
    # print(f"{doc.page_content} - {score}")
    samples.append(doc.page_content)

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

prompt = prompt_template.invoke({"samples": samples, "text": query})
print(prompt)

llm = ChatTongyi(
    model_name="qwen-plus",
    temperature=0.5,
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
)

print(llm.invoke(prompt))
