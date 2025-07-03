import os, datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic.v1 import BaseModel, Field

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

# 設定版本為LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"  # 設置LangSmith的項目名稱 LangSmith是LangChain中的子項目 用以項目數據追蹤
os.environ["LANGCHAIN_API_KEY"] = "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"

# 聊天機器人案例
# 創建模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)  # temperature影響多樣性 越高越隨機


class Classification(BaseModel):
    """
    決定怎麼個分類法 文本分類又稱為標簽 可以同時多種類型分類
    可以通過分值來定義情緒的程度 標籤

    定義一個Pydantic的數據類型 未來需要根據該類型 完成文本的分類
    """
    # 文本的情感傾向 預期為字符串類型
    sentiment: str = Field(description="文本的情感，可選值: happy, neutral, sad, angry")

    # 文本的攻擊性 預期為1到10的整數 沒有設置 *分數可能有0或大於9 因而需要規範
    aggressiveness: int = Field(
        description="描述文本的攻擊性，數字越大表示越有攻擊性(1-10)",
    )

    # 文本使用的語言 預期為字符串類型
    language: str = Field(description="文本使用的語言，可選值: Spanish, English, 中文, Italian")

# class Classification(BaseModel):
#     # 文本的情感傾向 預期為字符串類型
#     sentiment: str = Field(..., enum=['happy', 'neutral', 'sad'], description="文本的情感")
#
#     # 文本的攻擊性 預期為1到10的整數 沒有設置 *分數可能有0或大於9 因而需要規範
#     aggressiveness: int = Field(..., enum=[1, 2, 3, 4, 5],  # 只能給enum內的分數
#                                 description="描述文本的攻擊性，數字越大表示越有攻擊性",
#                                 )
#
#     # 文本使用的語言 預期為字符串類型
#     language: str = Field(..., enum=["Spanish", "English", "Chinese", "Italian"], description="文本使用的語言")


tagging_prompt = ChatPromptTemplate.from_template(
    """
    從以下段落中提取所需信息。
    只提取"Classification"類中提到的屬性。
    段落:
    {input}
    """
)

chain = tagging_prompt | model.with_structured_output(Classification)

input_text = "中國人民大學的王教授，道德敗壞，做出的事情實在讓我生氣!"
# input_text = "Hola, encantado de conocerte."
result: Classification = chain.invoke({"input": input_text})
print(result)
# 結果: sentiment='angry' aggressiveness=9 language='chinese'
# sentiment表示情感 aggressiveness表示生氣的級別 language表示語言 3個標籤

# 如果我不想用positive 或設定多個選項要如何 那麼便需要自己規註
