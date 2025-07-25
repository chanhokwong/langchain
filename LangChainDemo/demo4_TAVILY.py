import os

from langchain_chroma import Chroma
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import chat_agent_executor

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

# 設定版本為LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"  # 設置LangSmith的項目名稱 LangSmith是LangChain中的子項目 用以項目數據追蹤
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api"  # 前往LangSmith官網獲取

os.environ["TAVILY_API_KEY"] = "your_tavily_api"  # 前往Tavily官網獲取

# 聊天機器人案例
# 創建模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # 模型名字可從官網中查看

# 沒有任何工具的情況下 大模型會回答不知道
# result = model.invoke([HumanMessage(content="北京最近的天氣怎麼樣?")])
# print(result)

# LangChain內置了一個工具 可以輕鬆地使用Tavily搜索引擎作為工具
search = TavilySearchResults(max_results=2)  # 這個搜索引擎的工具會幫我們搜索互聯網中所有的網站 里面存了一些元數據 包括各種常用的公共網站 這里只想它返回兩個結果
# print(search.invoke("北京的天氣怎麼樣"))  # 調用invoke就知道它是一個runnable對象

# bind_tools 讓模型綁定工具
tools = [search]
# model_with_tools = model.bind_tools([search])

# 模型可以自動推理 是否需要調用工具去完成用戶的答案
# resp = model_with_tools.invoke([HumanMessage(content="中國的首都是哪個城市?")])
# 現在的情況有兩個 一個是大模型知道答案 因而它會直接回答 另一個則是大模型不知道答案 因而它會通過工具來獲取答案
# print(f"Model_Result_Content: {resp.content}")
# print(f"Tool_Result_Content: {resp.tool_calls}")  # 返回的結果為搜索的指令

# resp2 = model_with_tools.invoke([HumanMessage(content="北京最近的天氣怎麼樣?")])
# print(f"Model_Result_Content: {resp2.content}")
# print(f"Tool_Result_Content: {resp2.tool_calls}")

# 創建代理
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools=tools)

# message的key值不是固定的 key只是為了後面取值更容易
resp = agent_executor.invoke({"messages":[HumanMessage(content="甚麼是ai?")]})
print(resp["messages"])

resp2 = agent_executor.invoke({"messages":[HumanMessage(content="北京最近的天氣怎麼樣?")]})
print(resp2["messages"])

print(resp["messages"][2].content)
# 返回的信息有三類 HumanMessage AIMessage ToolMessage
# 如果未來在企業 只要做判斷就行了 如果第二個對象AIMessage的值為空 那麼便取這個值 resp["messages"][2].content
