from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

# 調用大模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # 需要指定model屬性

# 提示詞模板
prompt_template = ChatPromptTemplate.from_messages([
		# 給大模型的提示
    ("system", "你是一個樂於助人的助手，你將盡可能以{language}來回答所有的問答"),
    # MessagesPlaceholder用於後繼傳入問題 而問題的傳參變量為input
    MessagesPlaceholder(variable_name="input")
])

# 鏈 提示模板->大模型
chain = prompt_template | model

# 創建空字典 
# 用於儲存用戶與大模型的對話紀錄 讓大模型可以根據對話紀錄直接載入當時對話
store = {}

# 獲取歷史對話內容函數 傳參為對話的編號
def get_session_history(session_id: str):
		# 如果session_id不在儲存歷史對話的字典中 
    if session_id not in store:  
			  # 那麼便意昧著這是一個新的對話 所以把對話加到字典store當中
        store[session_id] = ChatMessageHistory()
    # 如果session_id存在 那麼返回對應session_id的歷史對話紀錄
    return store[session_id]

# 創建一個可讀取歷史紀錄的do_message主鏈 
do_message = RunnableWithMessageHistory(
    chain,  # 子鏈 可直接傳參大模型
    get_session_history,  # 獲取歷史對話內容函數
    input_messages_key="input"  # 設置傳參變量為input 以響應提示模板中的輸入變量
)

# 設置session_id的值 一般而言 這個值應是隨機生成 這個值就相當於儲存的名稱或編碼
config = {"configurable":{"session_id":"test1"}}  # 設置當前對話紀錄儲存的key值 即session_id的值

# 調用鏈 傳參
resp1 = do_message.invoke({
    "language":"中文",
    "input":[HumanMessage(content="你好，我的名字叫陳浩光")]  # 由於前面提示模板中只寫了傳參變量input 因而需要使用HumanMessage
},config=config)  # config的作用在於傳session_id的值 以儲存對話
print(resp1.content)

resp2 = do_message.invoke({
    "language":"中文",
    "input":[HumanMessage(content="我的名字叫甚麼?")]
},config=config)
print(resp2.content)
