"""
讓大模型識別語義 提取關鍵信息
"""

import os, datetime
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel
from pydantic.v1 import Field

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

# 設定版本為LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"  # 設置LangSmith的項目名稱 LangSmith是LangChain中的子項目 用以項目數據追蹤
os.environ["LANGCHAIN_API_KEY"] = "your_tracing_api_key"

# 聊天機器人案例
# 創建模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # 模型名字可從官網中查看

# 從一段文字中解析出我們想要的信息 我們的信息是節錄化的

# pydantic: 處理數據 驗證數據 定義數據格式 虛擬化和反虛擬化 類型轉換

# 定義一個數據模型 假如要提取一個人的信息
class Person(BaseModel):
    """
    關於一個人的數據模型
    """
    name: Optional[str] = Field(default=None, description="表示人的名字")

    hair_color: Optional[str] = Field(default=None, description="如果知道的話，這個人的頭發顏色")

    height_in_meters: Optional[str] = Field(default=None, description="以米為單位測量的高度")

# 幫助我們提取結構化的數據 數據可能不存在 所以設為可選的
# 會提取的信息和可能會提取的信息都可以放到數據模型上

class ManyPerson(BaseModel):
    """
    數據模型類: 代表多個人
    """
    people: List[Person]

# 提示模板 當有人輸入一段話 或 從文本中讀取的信息 都是要放到提示模板上
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一個專業的提取算法。"
         "只從未結構化文本中提取相關信息。"
         "如果你不知道要提取的屬性的值。"
         "返回該屬性的值為null。"
        ),
        # MessagePlaceholder('example')  # 如果需要上下文 可加上此代碼
        ("human", "{text}")
    ]
)

chain = {'text': RunnablePassthrough()} | prompt | model.with_structured_output(schema=ManyPerson)  # with_structured_output表示模型的輸出為一個結構化的數據

# text = '馬路上走來一個女生，長長的黑頭發披在肩上，大概1米7左右。'

# 只能提取一個 name='劉海' hair_color=None height_in_meters='1.8'
# 然而我們希望存取多個人
# text = '馬路上走來一個女生，長長的黑頭發披在肩上，大概1米7左右。走在她旁邊的是她的男朋友，叫:劉海:比她高10厘米'

# 除了中文 英文也行
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."

resp = chain.invoke(text)
print(resp)
