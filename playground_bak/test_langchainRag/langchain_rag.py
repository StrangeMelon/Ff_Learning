import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import PromptTemplate
from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv
from rich import print

# 1、加载环境变量
load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

# 2、实例化模型
chat = ChatZhipuAI(api_key=api_key, model="glm-4")

# 3、从网页中抓取和加载内容
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

# 4、使用RecursiveCharacterTextSpliter将内容分割成更小的块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_chunks = text_splitter.split_documents(docs)    # 结果是一个Document列表，会把docs分割成多个Document对象，每个Document对象中都有一个meta属性和一个page_content属性

# print(splits)
# for index_docs, doc_chunk in enumerate(doc_chunks):
#     # print(f"Document {index_docs + 1}")
#     print(doc_chunk)
#     # for index_field, field in enumerate(doc_chunk):
#     #     print(f"field {index_field + 1}: {field}")
#     print("\n" + "*"*60 + "\n")


# 5、对每个块，使用嵌入模型进行向量化
# client = ZhipuAI(api_key=api_key)

# embeddings = []
# for doc_chunk in doc_chunks:
#     for field_type, field_content in doc_chunk:  
#         if field_type == "page_content" and field_content.strip():
#             try:
#                 response = client.embeddings.create(
#                     model="embedding-3",
#                     dimensions=1024,
#                     input=field_content
#                 )
#             except Exception as e:
#                 print(f"请求失败，错误信息：{e}")
#             else:
#                 if hasattr(response, "data"):
#                     embeddings.append(response.data[0].embedding)

# 打印嵌入向量
# for index, embedding in enumerate(embeddings):
#     print(f"Embedding {index + 1}: {embedding[:3]}")
# print(len(embeddings[0]))

class EmbeddingGenerator:
    def __init__(self, model_name, dimensions):
        self.model_name = model_name
        self.dimensions = dimensions
        self.client = ZhipuAI(api_key=api_key)

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings.create(model=self.model_name, dimensions=self.dimensions, input=text)
            except Exception as e:
                print(f"请求失败，错误信息：{e}")
            else:
                if hasattr(response, "data"):
                    embeddings.append(response.data[0].embedding)
                else:
                    embeddings.append([0] * self.dimensions)
        return embeddings
    
    def embed_query(self, query):
        try:
            response = self.client.embeddings.create(model=self.model_name, dimensions=self.dimensions, input=query)
        except Exception as e:
            print(f"请求失败，错误信息：{e}")
            return [0] * 1024
        else:
            if hasattr(response, "data"):
                return response.data[0].embedding
            else:
                return [0] * 1024
                
embGenerator = EmbeddingGenerator(model_name="embedding-3", dimensions=1024)
texts = [field_content for doc_chunk in doc_chunks for field_type, field_content in doc_chunk if field_type=="page_content"]

# 6、创建 Chroma VectorStore， 并存入向量
chroma_store = Chroma(
    collection_name="example_collection",
    embedding_function=embGenerator,
    create_collection_if_not_exists=True,
)

IDs = chroma_store.add_texts(texts=texts)

print("Added documents with IDs: ", IDs)
print(len(IDs))

retriver = chroma_store.as_retriever()  # 将vectorstore转换为一个检索器，能够根据查询获取最相关的文本片段
prompt = """
you are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.If you don't know the answer, just say you don't know. Use three sentences maximum and keep the answer concise.
Question:{question}
Context:{context}
Answer:
        """ # 这里 'hub.pull' 是从某处获取提示的方法

prompt_runnable = PromptTemplate.from_template(prompt)

# 自定义函数format_docs 用于适当的格式化这些片段
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7、构建RAG链
rag_chain = (
    {"context": retriver | format_docs, "question": RunnablePassthrough()}
    | prompt_runnable
    | chat
    | StrOutputParser()
)

# 8、进行提问
rag_res = rag_chain.invoke("What is Task Decomposition?")
print(rag_res)

# 9、此命令指示 vectorstore 删除其保存的整个数据集合。这里的集合是指所有文档（文本片段）及其相应的已被索引并存储在向量存储中的向量表示的集合。
chroma_store.delete_collection()