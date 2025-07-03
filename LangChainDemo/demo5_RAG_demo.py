import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
import bs4

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

# 設定版本為LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"  # 設置LangSmith的項目名稱 LangSmith是LangChain中的子項目 用以項目數據追蹤
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api"  # 前往LangSmith官網獲取


# 聊天機器人案例
# 創建模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # 模型名字可從官網中查看

# 1. 加載數據 可以是本地文檔 可以是互聯網上的blog
# 加載數據: 一篇博客的內容
loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-header","post-title","post-content"))  # 如果僅寫post title 那麼僅會拿到title class_是類選擇器
    )
)  # 允許爬取一個或多個網頁 一個就字符串 多個就用列表

docs = loader.load()

# print(len(docs))
# print(docs)  # 返回一個Document

# 直接導入並不合適 上下文限制
# 2. 大文本的切割
# text = "Lilian Weng\n\n\nBuilding agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview#\nIn a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:\n\nPlanning\n\nSubgoal and decomposition: The agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.\nReflection and refinement: The agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps, thereby improving the quality of final results.\n\n\nMemory\n\nShort-term memory: I would consider all the in-context learning (See Prompt Engineering) as utilizing short-term memory of the model to learn.\nLong-term memory: This provides the agent with the capability to retain and re"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)  # chunk_size設置每個片段多少的字符 chunk_overlap設置可重覆的字符 它不會切開單詞
splits = splitter.split_documents(docs)  # 獲得一個切割的數組 根據docs的數據類型 選用對應的切割 字符串是split_text documents對象是split_documents
# for s in res:
#     print(s, end="***\n")

# 3. 存儲
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

# 4. 檢索器
retriever = vectorstore.as_retriever()

# 與AI模型整合


# 創建一個問題的模版 *基於提示模版是英文的 因而最好用英文去問問題
system_prompt = """ You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. If
you don't know the answer, say that you don't know. Use three sentences maximum
and keep the answer concise. \n

{context}
"""

prompt = ChatPromptTemplate.from_messages(  # 提問和回答的歷史紀錄模板
    [
        ("system",system_prompt),  # 提示模板
        MessagesPlaceholder("chat_history"),  # 提問歷史紀錄 自定義變量chat_history
        ("human","{input}")
    ]
)

# 得到chain 通過函數來獲得chain
# chain1 可作問答
chain1 = create_stuff_documents_chain(model, prompt)  # 創建一個多文本的chain

# chain2 在問答的基礎上可作檢索
# chain2 = create_retrieval_chain(retriever, chain1)

# resp = chain2.invoke({"input":"What is Task Decomposition?"})
# print(resp["answer"])

"""
注意:
一般情況下，我們構建的鏈(chain)直接使用輸入回答紀錄來關聯上下文。但在此案例中，查詢檢索器也需要對話上下文才能被理解。

解決方法:
添加一個子鏈，它采用最新用戶問題和聊天歷史，并在它引用歷史信息中的任何信息時重新表述問題。這可以被簡單地認為是構建
這個子鏈的目的: 讓檢索過程融入對話的上下文
"""

# 創建一個子鏈
# 子鏈的提示模板
contextualize_q_system_prompt = """Given a chat history and the latest user question
 which might reference context in the chat history,
 formulate a standalone question which can be understood
 without the chat history. Do NOT answer the question, 
 just reformulate it if needed and otherwise return it as is."""

# 這個的目的是讓檢索器擁有理解上下文的能力
retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ('system',contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),  # 傳歷史紀錄就用MessagePlaceholder
        ("human","{input}")
    ]
)

# 創建一個子鏈
history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

# 保持問答的歷史紀錄
store = {}

def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 創建一個父鏈chain 把前兩個鏈整合
chain = create_retrieval_chain(history_chain, chain1)

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",  # 這個是因應human輸入的變量名稱而定的
    history_messages_key="chat_history",  # 這個是根據上面儲存歷史變量的名稱而定的
    output_messages_key="answer"  # answer是自定義變量 這個可以自定義
)

# 第一輪對話
resp1 = result_chain.invoke(
    {"input":"What is Task Decomposition?"},
    config={'configurable':{'session_id':'zs123456'}}
)

print(resp1["answer"])  # 這是根據output_messages_key定義的變量名稱來取值的


# 第二輪對話
resp2 = result_chain.invoke(
    {"input":"What are common ways of doing it?"},
    config={'configurable':{'session_id':'ls123456'}}
)

print(resp2["answer"])  # 這是根據output_messages_key定義的變量名稱來取值的
