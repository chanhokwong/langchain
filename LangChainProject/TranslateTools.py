# 教如何調用ai模型
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 設置API KEY (如果在環境變量中設置了API KEY 那麼便不用添加)
# os.environ["GOOGLE_API_KEY"] = "your_api_key"

# 設定版本為LANGCHAIN_TRACING_V2
# LANGCHAIN_TRACING_V2 用於追縱API使用的一些參數 如token數
# API_KEY從Langchain官網中獲取
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_Langchain_tracing_api"

# 調用大語言模型
# 1. 創建模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # 模型名字可從官網中查看

# 2. 準備prompt
msg = [
    SystemMessage(content="請將以下的內容翻譯成意大利語"),  # 模型提示
    HumanMessage(content="你好，請問你要去哪里?"),  # 你的問題
]

# 簡單的解析響應數據 (僅獲得大模型的回答)
# 3. 創建返回的數據解析器
parser = StrOutputParser()  # 文本解析器

# 定義提示模板 (輸入給大模型的模板)
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '請將下面的內容翻譯成{language}'),
    ('user', '{text}'),
])

# 4. 得到鏈
chain = prompt_template | model | parser

# 5. 直接使用chain來調用
# print(chain.invoke({'language': 'English', 'text': '我下午還有一節課,不能去打球了。'}))

# 要把我們的程序部署成服務
# 創建fastAPI的應用
# title是你應用的名稱 version是你服務的版本(可自定義)
app = FastAPI(title="我的Langchain服務", version="v1.0", description="使用Langchain翻譯任何語句的服務器")  # 這步是創建了fastapi的應用

# 添加路由
add_routes(
    app,
    chain,
    path="/chain"  # 路由可以隨便改
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002)
