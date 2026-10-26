# 把对联数据保存到Redis向量数据库中
from dotenv import load_dotenv
import redis
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()
embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

# 保存向量数据库
redis_url = "redis://127.0.0.1:6379"

redis_client = redis.from_url(redis_url)
# print(redis_client.ping()) # 测试连接 返回True表示连接成功

from langchain_redis import RedisVectorStore, RedisConfig

config = RedisConfig(
    index_name = "couplet",
    redis_url = redis_url
)

vector_store = RedisVectorStore(embedding_model, config=config)

lines = []
with open("../resources/test.csv", "r", encoding="utf-8") as file:
    for line in file:
        print(line)
        lines.append(line)
vector_store.add_text(lines)

