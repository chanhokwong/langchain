import os
from operator import itemgetter

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

# 設定版本為LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"  # 設置LangSmith的項目名稱 LangSmith是LangChain中的子項目 用以項目數據追蹤
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api"  # 前往LangSmith官網獲取

# 聊天機器人案例
# 創建模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # 模型名字可從官網中查看

# sqlalchemy
# dialect+driver://username:password@host:port/database
# example: engine = create_engine('sqlite:///C:\\path\\to\\foo.db')
# 默認不用+
# example mysql: engine = create_engine('mysql+mysqldb://scott:tiger@localhost/foo?charset=utf8')

# 初始化MySQL數據庫的連接
HOSTNAME = '127.0.0.1'
PORT = "3306"
DATABASE = 'your_db'  # 填寫你的db名稱
USERNAME = 'root'  # 填寫你的db帳號
PASSWORD = 'password'  # 填寫你的db密碼

# sqlalchemy 驅動URL
MYSQL_URI = f'mysql+mysqldb://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4'

db = SQLDatabase.from_uri(MYSQL_URI)

# 測試連接是否成功
# get_usable_table_names() => 拿到可以使用的表
# print(db.get_usable_table_names())
# print(db.run('SELECT * FROM employee limit 10;'))  # limit是限制返回數據條數

# 直接使用大模型和數據庫整合 只能根據你的問題生成SQL 但不會直接執行
# test_chain = create_sql_query_chain(model, db)
# resp = test_chain.invoke({"question":"請問:員工表有多少條數據?"})  # 傳參是一個字典
# print(resp)  # resp儲存的為sql指令
# ```sql
# SELECT count(*) FROM employee
# ```


answer_prompt = PromptTemplate.from_template(
    """ 給定以下用戶問題、SQL語句和SQL執行後的結果，回答用戶問題。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答: """
)

# 創建一個執行sql語句的工具
execute_sql_tool = QuerySQLDataBaseTool(db=db)

# 初始化生成SQL的chain
sql_chain = create_sql_query_chain(model, db)

# 1. 生成SQL語句  2. 執行SQL
# 2. 模板
chain = (RunnablePassthrough.assign(query=sql_chain).assign(result=itemgetter("query") | execute_sql_tool)
         | answer_prompt
         | model
         | StrOutputParser()
         )

resp = chain.invoke(input={"question":"請問:員工表有多少條數據?"})
print(resp)
