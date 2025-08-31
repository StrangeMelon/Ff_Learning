import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

Base = declarative_base()

class ChatSession(Base):
    """ORM model: sessions table (chat session metadata)
    Use ChatSession to avoid name collision with SQLAlchemy session factory."""
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")

class Message(Base):
    """ORM model: messages table"""
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("ChatSession", back_populates="messages")

class Config:
    def __init__(self):
        self.api_key = os.getenv("ZHIPU_API_KEY")
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///chat_history.db")
        self.collection = os.getenv("CHROMA_COLLECTION", "example_collection")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "embedding-3")
        self.embedding_dims = int(os.getenv("EMBEDDING_DIMS", "1024"))
        self.llm_model = os.getenv("LLM_MODEL", "glm-4")

# 定义会话持久化存储的类
class DBHistoryStore:
    """Minimal DB persistence wrapper for ChatMessageHistory using SQLAlchemy."""
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def save_message(self, session_id: str, role: str, content: str):
        db = self.SessionLocal()
        try:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if not session:
                session = ChatSession(session_id=session_id)
                db.add(session)
                db.commit()
                db.refresh(session)
            db.add(Message(session_id=session.id, role=role, content=content))
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise
        finally:
            db.close()
    
    def load_session_history(self, session_id: str) -> ChatMessageHistory:
        db = self.SessionLocal()
        history = ChatMessageHistory()
        try:
            s = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if s:
                for m in s.messages:
                    # ChatMessageHistory.add_message accepts dict-like messages
                    history.add_message({"role": m.role, "content": m.content})
        finally:
            db.close()
        return history

class EmbeddingClient:
    def __init__(self, api_key: str, model_name: str, dims: int):
        self.client = ZhipuAI(api_key=api_key)
        self.model_name = model_name
        self.dimensions = dims
    
    def embed_documents(self, texts):
        result = []
        for text in texts:
            try:
                response = self.client.embeddings.create(model=self.model_name, input=text, dimensions=self.dimensions)
            except Exception as e:
                print(f"请求失败，错误信息：{e}")
                result.append([0] * self.dimensions)
            else:
                if hasattr(response, "data") and response.data:
                    result.append(response.data[0].embedding)
                else:
                    result.append([0] * self.dimensions)
        return result
     
    def embed_query(self, query):
        try:
            response = self.client.embeddings.create(model=self.model_name, input=query, dimensions=self.dimensions)
        except Exception as e:
            print(f"请求失败，错误信息：{e}")
            return [0] * self.dimensions
        else:
            if hasattr(response, "data") and response.data:
                return response.data[0].embedding
            else:
                return [0] * self.dimensions

class VectorStoreWrapper:
    def __init__(self, collection_name: str, embedding_function):
        # embedding_function can be an object with embed_documents or a callable
        self.store = Chroma(collection_name=collection_name, embedding_function=embedding_function, create_collection_if_not_exists=True)

    def add_texts(self, texts):
        return self.store.add_texts(texts=texts)    
    
    def as_retriever(self):
        return self.store.as_retriever()

class RAGBuilder:
    @staticmethod
    def build_contextualize_prompt():
        prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
        return ChatPromptTemplate(
        [
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
        )

    @staticmethod
    def build_qa_prompt():
        qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"""
        return ChatPromptTemplate([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

class ConversationService:
    def __init__(self, rag_chain, history_loader_callable, db_store: DBHistoryStore):
        self.rag_chain = rag_chain
        self.history_loader_callable = history_loader_callable
        self.db_store = db_store
        # internal in-memory cache for active session histories
        self.store = {}

    def get_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = self.history_loader_callable(session_id)
        return self.store[session_id]

    def save_and_invoke(self, session_id: str, input_text: str):
        # persist user message first
        self.db_store.save_message(session_id=session_id, role="human", content=input_text)
        # invoke rag chain
        result = self.rag_chain.invoke(
            {"input": input_text}, 
            config={"configurable": {"session_id": session_id}}
        )["answer"]
        # persist ai answer
        self.db_store.save_message(session_id=session_id, role="ai", content=result)
        
        return result

if __name__ == '__main__':
    cfg = Config()
    # print(cfg.api_key)
    # print(cfg.collection)
    # print(cfg.database_url)
    # print(cfg.embedding_dims)
    # print(cfg.embedding_model)
    # print(cfg.llm_model)
    chat = ChatZhipuAI(api_key=cfg.api_key, model=cfg.llm_model, temperature=0.5)

    db_store = DBHistoryStore(cfg.database_url)

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_chunks = splitter.split_documents(docs)

    texts = [doc.page_content for doc in doc_chunks]
    
    emb = EmbeddingClient(cfg.api_key, cfg.embedding_model, cfg.embedding_dims)
    chroma = VectorStoreWrapper(collection_name=cfg.collection, embedding_function=emb)
    IDs = chroma.add_texts(texts=texts)

    retriever = chroma.as_retriever()

    contextualize_prompt = RAGBuilder.build_contextualize_prompt()

    history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_prompt)

    qa_prompt = RAGBuilder.build_qa_prompt()
    question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        db_store.load_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    service = ConversationService(conversational_rag_chain, db_store.load_session_history, db_store)

    first_qa = service.save_and_invoke("abc123", "what is Task Decomposition?")
    second_qa = service.save_and_invoke("abc123", "What are common ways of doing it?")
    print(f"first_qa: {first_qa}")
    print(f"second_qa: {second_qa}")
    