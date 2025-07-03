import os, datetime

from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import CharacterTextSplitter
from pydantic.v1 import BaseModel, Field

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

# 設定版本為LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"  # 設置LangSmith的項目名稱 LangSmith是LangChain中的子項目 用以項目數據追蹤
os.environ["LANGCHAIN_API_KEY"] = "your_tracing_api_key"

# 聊天機器人案例
# 創建模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)  # temperature影響多樣性 越高越隨機

# 加載我們的文檔 我們將使用WebBaseLoader來加載博客文章
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()  # 得到整篇文章

# 第三種 Refine
"""
Refine: RefineDocumentsChain 類似於map-reduce
文檔鏈通過循環遍歷輸入文檔並逐步更新其答案來構建響應。對於每個文檔，它將當前文檔和最新的中間答案傳递給LLM鏈，以獲得新的答案。
切成一小塊一小塊 D1,D2,...,D5000 拿著D1給大模型 得到一個摘要S1 然後再把S1和D2傳給大模型 得到一個新的S1 再把D3和新的S1再給大模型 以獲得一個新的S1 以此類推
速度一定沒有MAP REDUCE快 解決了DOCUMENT超過了大模型一次性TOKEN的問題
"""
# 第一步: 切割階段
# 每一個小docs為1000個token
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)  # chunk_size是每個文本的token上限
split_docs = text_splitter.split_documents(docs)

# 指定chain_type為: refine
chain = load_summarize_chain(model,chain_type="refine")

result = chain.invoke(split_docs)
print(result["output_text"])  # "output_text"這個是固定參數
