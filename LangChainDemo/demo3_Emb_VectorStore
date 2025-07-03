from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import time

# 調用模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 設置文檔數據 這里有五篇文章
documents = [  # 由五篇文章組成的列表
    Document(
        page_content="狗是偉大的伴侶，以其忠誠和友好而聞名。",  # page_content是文本內容
        metadata={"source": "哺乳動物寵物文檔"} ),  # metadata是元數據庫 作者信息、來源、摘要等 鍵值可以隨便設置 無需強制source
    Document(
        page_content="貓是獨立的寵物，通常喜歡自己的空間。",
        metadata={"source": "哺乳動物寵物文檔"} ),
    Document(
        page_content="金魚是初學者的流行寵物，需要相對簡單的護理。",
        metadata={"source": "魚類寵物文檔"} ),
    Document(
        page_content="鸚鵡是聰明的鳥類，能夠模仿人類的語言。",
        metadata={"source": "鳥類寵物文檔"} ),
    Document(
        page_content="免子是社交動物，需要足夠的空間跳躍。",
        metadata={"source": "哺乳動物寵物文檔"} ),
]

# 調用嵌入模型
embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

# 向量數據庫 向量化儲存文檔數據 
# 處理邏輯: 將文檔數據傳到嵌入模型進行處理 然後儲存到向量數據庫中
# *需要注意from_documents只是因應documents object的檔案傳入 如果是其他檔案需要更換
# vector_store = Chroma.from_documents(documents=documents, embedding=embedding)

# 批次向量化儲存文檔數據
# 由於嘗試將大量文件(documents)轉換為向量(embeddings)，因而其將過於頻繁地呼叫了大模型的API，從而造成錯誤
# 因而為了防止錯誤發生，將採用下列方法將量化批次化
batch_size = 100
# 先用第一個批次創建 Vector Store
first_batch = documents[:batch_size]
print("正在處理第一個批次...")
vector_store = Chroma.from_documents(documents=first_batch, embedding=embedding)
print("第一個批次處理完成。")

# 處理剩餘的批次
for i in range(batch_size, len(documents), batch_size):
    # 加上延遲，避免請求過於頻繁
    print("等待 2 秒...")
    time.sleep(2)  # 等待 2 秒

    # 取得下一個批次
    batch = documents[i:i + batch_size]
    print(f"正在處理批次 {i // batch_size + 1} (文件索引 {i} 到 {i + len(batch) - 1})...")

    # 將新的文件批次添加到已有的 Vector Store 中
    vector_store.add_documents(documents=batch)
    print(f"批次 {i // batch_size + 1} 處理完成。")


# 使用關鍵字[狸花貓]來獲取其相似度分數
# 想要相似度分數便使用.similarity_search_with_score
# 如果不需要相似度分數，僅獲得其相似度參數，使用.similarity_search就好
# print(vector_store.similarity_search_with_score("狸花貓"))

# 檢索器
# 對相似度參數進行檢索，僅存取相似度最高的數據
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)

# 提示模板語句
# 存在兩個參數 question context
prompt_template = """
你將根據上下文來回答下列問題:
{question}
上下文:
{context}
"""

# 提示模板
# 設定身份為用戶並添加提示模板語句
prompt = ChatPromptTemplate.from_messages([("human", prompt_template)])

# 鏈
# 將問題和文本檢索參數傳到提示模板中，然後傳到大模型中
# RunnablePassthrough允許後面用戶傳參，但需注意加上()
# question為問題傳參 context為檢索對象 傳入檢索對象會進行相似度檢索
chain = {"question":RunnablePassthrough(),"context":retriever} | prompt | model

# 調用鏈，詢問問題
resp = chain.invoke("請介紹一下貓?")
print(resp.content)
