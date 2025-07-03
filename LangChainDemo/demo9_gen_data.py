import os, json
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# 你的環境變量設置（保持不變）
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
# os.environ["LANGCHAIN_API_KEY"] = "your_tracing_api_key"

# --- 步驟 1: 更新數據模型 ---
# 使用標準的 Pydantic V2 BaseModel
# 為字段添加描述 (description)，這會作為給大模型的提示，能極大提高生成質量
class MedicalBilling(BaseModel):
    """單個醫療賬單的詳細信息。"""
    patient_id: int = Field(..., description="患者的唯一數字ID")
    patient_name: str = Field(..., description="患者的全名")
    diagnosis_code: str = Field(..., description="國際疾病分類（ICD-10）診斷代碼, 例如 J20.9")
    procedure_code: str = Field(..., description="醫療程序代碼, 例如 99203")
    total_charge: float = Field(..., description="本次賬單的總費用")
    insurance_claim_amount: float = Field(..., description="向保險公司索賠的金額")

# 為了讓模型一次性生成一個列表，我們定義一個包含列表的模型
class MedicalBillings(BaseModel):
    """包含多個醫療賬單記錄的列表。"""
    bills: List[MedicalBilling]

# --- 步驟 2: 創建模型 ---
try:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)
except Exception as e:
    print(f"初始化 gemini-1.5-flash 失敗: {e}")
    print("嘗試使用 gemini-1.0-pro...")
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.8)


# --- 步驟 3: 創建一個支持結構化輸出的鏈 ---
# 將 Pydantic 模型與 LLM 綁定
structured_llm = model.with_structured_output(MedicalBillings)

# --- 步驟 4: 創建新的提示模板 ---
# 這個模板更直接地告訴模型要做什麼
# 我們將樣例數據直接放在提示中作為指導
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一個專業的合成數據生成器。你的任務是根據用戶要求，創建高質量的、符合指定格式的醫療賬單數據。
            請嚴格遵循用戶提供的 Pydantic 類結構來生成數據。
            這裡有一些數據樣例，你可以參考其風格和格式：
            - Patient ID: 123456, Patient Name: 張娜, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: 500.0, Insurance Claim Amount: 3000.0
            - Patient ID: 789012, Patient Name: 王興鵬, Diagnosis Code: M54.5, Procedure Code: 99213, Total Charge: 150.0, Insurance Claim Amount: 2000.0
            - Patient ID: 345678, Patient Name: 劉曉輝, Diagnosis Code: E11.9, Procedure Code: 99214, Total Charge: 300.0, Insurance Claim Amount: 4000.0
            """,
        ),
        ("human", "請為我生成 {count} 條關於 `{subject}` 的數據。{extra}"),
    ]
)

# --- 步驟 5: 構建並調用鏈 ---
# 將提示和帶有結構化輸出功能的 LLM 組合成一個鏈
chain = prompt | structured_llm

# 調用鏈，傳入參數
result = chain.invoke({
    "count": 10,
    "subject": "醫療賬單",
    # "extra": "名字可以是隨機的，請盡量使用一些不常見的中文人名。",
    "extra": "醫療總費用呈現正態分佈，最小的總費用為1000"
})

# 打印結果，result現在直接就是 MedicalBillings 類的實例
# 我們可以將其轉換為字典列表以便更好地打印
pretty_result = [bill.dict() for bill in result.bills]
print(json.dumps(pretty_result, indent=2, ensure_ascii=False))

# 生成數據後 只需用大模型生成數據庫指令 就可以把數據傳入數據庫中了
