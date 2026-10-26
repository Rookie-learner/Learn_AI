from langchain_classic.chains.retrieval import create_retrieval_chain

FILE_PATH = "./政策文件.txt"


# 收集数据
def load_file(file_path):
    from langchain_community.document_loaders import TextLoader
    # 加载文件
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents


# 文本分块
def cut_file_content(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,  # 适合中文段落
        chunk_overlap=0,  # 重叠0个字符，避免上下缺失
        separators=['\n\n', '\n', '。', '！', '？']  # 段落分隔符
    )
    splits = text_splitter.split_documents(documents)

    print(f"分块后的分块数量: {len(splits)}")
    # 打印分块
    # print(f"分块后的第一个分块: {splits[0].page_content}")
    # for split in splits:
    #     print("-----------------")
    #     print(split.page_content)

    return splits


# 选择文本嵌入模型
def choose_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    # 加载本地模型
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="bge-base-zh-v1.5",
    #     model_kwargs={"device": "cpu"},
    #     encode_kwargs={"normalize_embeddings": False}, # 不归一化向量，提升相似度计算效率
    # )
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embeddings


# 初始化向量数据库
def init_vector_db(splits, embeddings):
    from langchain_chroma import Chroma
    # 初始化向量数据库
    db = Chroma.from_documents(
        collection_name="policy_db",
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    return db


# 结合LLM模型使用
def RAG_chain(db):
    # 加载LLM模型
    # DeepSeek
    import os
    from langchain_deepseek import ChatDeepSeek
    key = "sk-xxxxxxxxxxxxxxxxxxxxxx"
    os.environ["DEEPSEEK_API_KEY"] = key
    llm = ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0.5,
    )

    # 创建一个RAG链
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
             "你是一个专业的政策问答机器人，你的任务是根据上下文{context}回答用户的问题。带有一定的解释能力，如果你不知道答案，就说你不知道。"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    chain = create_retrieval_chain(db.as_retriever(), question_answer_chain)

    return chain


if __name__ == "__main__":
    documents = load_file(FILE_PATH)
    print("加载文件完成")
    splits = cut_file_content(documents)
    print("分块完成")
    embeddings = choose_embeddings()
    print("加载嵌入模型完成")
    db = init_vector_db(splits, embeddings)
    print("初始化向量数据库完成")
    # 查询数据库
    # query = "年假的有效期"
    query = "老板的名字"
    # results = db.similarity_search(query, k=1)
    # print("查询结果:")
    # for result in results:
    #     print(result.page_content)

    # 使用RAG链查询
    rag_chain = RAG_chain(db)
    print("RAG链创建完成")
    print("开始查询...")
    result = rag_chain.invoke({"input": query})
    print("查询结果: ", result["answer"])
    print("查询文档: ", result["context"][0].page_content)
