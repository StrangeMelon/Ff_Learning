
"""
使用 SQLAlchemy 在 SQLite 数据库中保存聊天历史记录
"""
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

# 
class Session(Base):
    """
    Seesion表示管理聊天对话的表
    """
    __tablename__ = "sessions"
    id =  Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")

class Message(Base):
    """
    Message用来表示每个会话中的消息
    """
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")

# def Load_db(url):
url = None
engine = create_engine(url)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)



def get_db():
    """
    创建一个生成器用来管理数据库会话，该函数将确保每个数据库会话能正常打开与关闭
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_message(session_id: str, role: str, content: str):
    """
    用于将各个消息保存到数据库中。该函数会检查会话是否存在，如果不存在，则创建一个会话，然后把消息保存到这个会话中
    """
    db = next(get_db())
    try:
        # 检查会话是否存在，如果不存在就创建一个会话；如果会话存在，就把消息保存在这个会话
        session = db.query(Session).filter(Session.session_id==session_id).first()
        if not session:
            # 创建新会话
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)
        # 执行消息保存逻辑
        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()

# load_session_history 一次只能加载一个会话的历史消息
def load_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    从数据库中加载指定的聊天历史。此函数会根据给定的session_id检索Session表，然后将所有与这个session_id相关联的message全部都取出来，构建成聊天记录
    """
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        session = db.query(Session).filter(Session.session_id==session_id).first()
        if session:
            # 会话存在，就把会话绑定的消息全都取出来
            for message in session.messages:
                chat_history.add_message({"role": message.role, "content": message.content})
    except SQLAlchemyError:
        pass
    finally:
        db.close()
    return chat_history

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    此函数用于调用load_session_history从数据库中加载指定session_id的会话的历史消息并返回给外界
    如果这个对话本来就在store中，那么直接返回
    如果这个对话不在store中，就调用load_session_history从数据库中取出历史消息，并且添加在store中
    """
    if session_id not in store: # store是一个缓存，用来存放当前的多个会话的历史消息，是一个字典，每一个元素的value对应的是一个ChatMessageHistory()对象
        store[session_id] = load_session_history(session_id)
    return store[session_id]

def save_all_sessions():
    """
    在退出应用程序之前保存所有的会话功能。此函数迭代内存中的所有会话并将其消息保存到数据库中
    增加错误处理以确保程序稳定性
    """
    from langchain_core.messages import HumanMessage, AIMessage

    for session_id, chat_history in store.item():
        for message in chat_history.messages:
            if isinstance(message, HumanMessage):
                save_message(session_id=session_id, role="human", content=message.content)
            elif isinstance(message, AIMessage):
                save_message(session_id=session_id, role="ai", content=message.content)
            elif isinstance(message, dict):
                if "role" in message and "content" in message:
                    save_message(session_id=session_id, role=message["role"], content=message["content"])
                else:
                    print(f"Skipped a message due to missing keys: {message}")
            else:
                print(f"kipped an unsupported message type: {message}")

