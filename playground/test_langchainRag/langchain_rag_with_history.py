import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.chat_message_histories import ChatMessageHistory
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from rich import print
from typing import List
import os

load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

chat = ChatZhipuAI(
    api_key=api_key,
    model="glm-4.5",
    temperature=0.5,
)   # 对话模型


# 加载数据
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_chunks = text_splitter.split_documents(docs)

# 获取每一个文本块
texts = [field_content for doc_chunk in doc_chunks for field_type, field_content in doc_chunk if field_type=="page_content"]

# 构建用于Chroma的自定义嵌入模型
class EmbeddingGenerator:
    def __init__(self, model_name, dimensions):
        self.model_name = model_name
        self.dimensions = dimensions
        self.client = ZhipuAI(api_key=api_key)
    
    def embed_documents(self, texts:List[str]):
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings.create(model=self.model_name, dimensions=self.dimensions, input=text)
            except Exception as e:
                print(f"请求失败，错误信息：{e}")
                embeddings.append([0] * self.dimensions)
            else:
                if hasattr(response, "data") and response.data:
                    embeddings.append(response.data[0].embedding)
                else:
                    embeddings.append([0] * self.dimensions)
        return embeddings

    def embed_query(self, query):
        try:
            response = self.client.embeddings.create(model=self.model_name, dimensions=self.dimensions, input=query)
        except Exception as e:
            print(f"请求失败，错误信息：{e}")
            return [0] * self.dimensions
        else:
            if hasattr(response, "data") and response.data:
                return response.data[0].embedding
            else:
                return [0] * self.dimensions

embGenerator = EmbeddingGenerator(model_name="embedding-3", dimensions=1024)

# 构建向量数据库
chroma_store = Chroma(
    collection_name="example_collection",
    embedding_function=embGenerator,
    create_collection_if_not_exists=True
)

# 将获得的文本块嵌入数据库，并返回每一个文本块的id
IDs = chroma_store.add_texts(texts=texts)   # IDs是一个字符串列表，每个字符串代表一个文本块的id

# 使用chroma_store创建检索器
retriver = chroma_store.as_retriever()

# 构建子链

# 对用户输入的query进行重构，生成用于查询数据库的query_db
contextualize_query_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_query_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_query_system_prompt),
        MessagesPlaceholder("chat_history"), # 历史消息的占位符
        ("human", "{input}")    # 这里的input就是用户输入的初始query
    ]
)

# 构设置历史信息感知检索器
# create_history_aware_retriever 函数旨在接受输入和“chat_history”的键，
# 用于创建集成聊天历史记录以进行上下文感知处理的检索器。
"""
如果历史记录存在, 就会把历史记录放在contextualize_query_prompt的MessagesPlaceholder("chat_history")的位置
然后再把用户的输入放在{input}位置, 生成一个有效的组合提示词contextualize_query_prompt
把这个提示词输入LLM(这里是chat), 之后LLM生成一个用于查询数据库的query
最后把这条query传入检索器(这里是retriver)
"""
history_aware_retriever = create_history_aware_retriever(
    chat, retriver, contextualize_query_prompt
)
# 至此，子链构建结束

# 自定义函数format_docs 用于适当的格式化这些片段
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Step 12. 定义 QA 系统的提示模板，指定系统应如何根据检索到的上下文响应输入。
# 该字符串设置语言模型的指令，指示它使用提供的上下文来简洁地回答问题。如果答案未知，则指示模型明确说明这一点。
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise. \

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")   # 这里的input就是重构后的query
    ]
)

question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 构建历史记录管理器
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
    
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

first_answer = conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable" : {"session_id": "abc123"}
    },
)["answer"]

second_answer = conversational_rag_chain.invoke(
    {"input":"What are common ways of doing it?"}, 
    config={
        "configurable": {"session_id": "abc123"}
    },
)["answer"]

from rich import print
print(f"first_answer: {first_answer}")
print("*" * 60)
print(f"second_answer: {second_answer}")
for chat_history in store.values():
    print(chat_history)
