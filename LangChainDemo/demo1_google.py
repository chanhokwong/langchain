from google import genai
import os

# ========================Gemini API KEY SETTING===============================

# 存取儲存在環境變量中的API Gemini的api_key便是GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")
# print(api_key)

# 如不把API儲存至環境變量 也可以在這里直接傳
# client = genai.Client(api_key="your api key")

# 定義客服端
client = genai.Client()

# 調用API 需指定模型類別 可從官方文檔查閱 https://ai.google.dev/gemini-api/docs/models?hl=zh-tw
# Gemini輸入內容的變量為contents 而其他的大模型可能是messages 這需要查閱對應的文檔
response = client.models.generate_content(
    model='gemini-2.0-flash',  # 選用模型的名稱
    contents='Tell me a story in 300 words.'  # 輸入詢問大模型的內容
)
print(response)  # 直接輸出 會有很多額外的信息 儲如模型的版本信息,token數等等
print(response.text)  # 從輸出中僅存取儲存回答的text變量

# print(response.model_dump_json(exclude_none=True, indent=4))





