import os, datetime
from typing import Optional, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic.v1 import BaseModel, Field

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

# 設定版本為LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"  # 設置LangSmith的項目名稱 LangSmith是LangChain中的子項目 用以項目數據追蹤
os.environ["LANGCHAIN_API_KEY"] = "your_tracing_api"

# 聊天機器人案例
# 創建模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # 模型名字可從官網中查看
# OpenAIEmbeddings: text-embedding-3-small
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

persist_dir = 'chroma_data_dir'  # 存放向量數據庫的目錄

# 一些Youtube的視頻連接
urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
]

# docs = []  # Document的數組
# for url in urls:
#     # 一個youtube的視頻對應一個document
#     # docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())
#     docs.extend(YoutubeLoader.from_youtube_url(url).load())
#
# # print(len(docs))
# # print(docs[0])
# # # 給docs添加額外的元數據: 視頻發布的年份
# #
# # for doc in docs:
# #     doc.metadata["publisher_year"] = int(
# #         datetime.datetime.strptime(doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S").strftime("%Y")
# #     )
# #
# # print(docs[0].metadata)
# # print(docs[0].page_content[:500])  # 第一個視頻的字幕內容
#
# # 根據多個doc構建向量數據庫
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)  # 2000字符 30可重覆字符
# split_doc = text_splitter.split_documents(docs)
#
# # 向量數據庫的持久化 需要第三個參數 持久化的好處就是不用再調用api來獲取數據 然後做處理 只需要去對應目錄存取即可
# vectorstore = Chroma.from_documents(split_doc,embedding,persist_directory=persist_dir)  # 并且把向量數據庫持久化到磁盤
#
# # chroma_data_dir 0a2a6cda文件夾里面的bin文件是儲存數據的

# 加載磁盤中的向量數據庫
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)

# 測試向量數據庫的相似檢索
result = vectorstore.similarity_search_with_score("how do i build a RAG agent")
# print(result)
# 結果
# [(Document(id='ed8a6b4a-a9d8-48d7-b642-b568f24937c3',
# metadata={'source': 'hvAPnpSfSGo'},
# page_content="..."),
# 0.6673682928085327)]

# print(result[0])
# print(result[0][0].metadata["source"])

system = """You are an expert at converting user questions into database queries.
You have access to a database of tutorial videos about a software library for building LLM powered application.
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}")
    ]
)

# pydantic 數據管理的一個庫 數據驗證、管理、定義、排序化
# 數據模型 web中的數據庫模型
class Search(BaseModel):
    """
    定義了一個數據模型
    整體過程: 獲取用戶的question內容 接著以Search模型對向量數據庫里面進行搜索
    """
    # 內容的相似性和發佈年份
    query: str = Field(None, description="Similarity search query applied to video transcipts")  # 相似度搜索
    # publish_year: Optional[int] = Field(None, description="Year video was published")  # Optional表示可選 即可對年份進行搜索 也可以不搜索
    source: Optional[int] = Field(None, description="Video source")


chain = {"question": RunnablePassthrough()} | prompt | model.with_structured_output(Search)

# resp1 = chain.invoke("How do I build a RAG agent?")  # 解析問題的結構
# print(resp1)  # resp是經過數據模型翻譯出來的內容
# 結果: 一個結構化的檢索條件
# query='RAG agent' publish_year=None
# resp2 = chain.invoke("videos on RAG published in 2023")  # 解析問題的結構
# print(resp2)
# 結果: 一個結構化的檢索條件
# query='RAG' publish_year=2023

# 它是如果能夠智能理解我們的需求 這是因為大模型、數據模型

def retrieval(search: Search)-> List[Document]:
    _filter = None
    if search.source:
        # 根據publish_year 存在得到一個檢索條件
        # "#$eq"是Chroma向量數據庫的固定語法
        _filter = {"publish_year": {"#$eq": search.publish_year}}
        # _filter = {"source": {"#$eq": search.source}}  # 這個是我自定義的
    return vectorstore.similarity_search(search.query, filter=_filter)

new_chain = chain | retrieval

result = new_chain.invoke("videos on RAG published in 2023")
# result = new_chain.invoke("RAG tutorial")
print([(doc.metadata['title'], doc.metadata['publish_year']) for doc in result])
# print([(doc.page_content) for doc in result])
