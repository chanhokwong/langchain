import csv
from typing import Type, Optional
import requests
import uvicorn
from fastapi import FastAPI
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import chat_agent_executor
from langserve import add_routes
from pydantic import BaseModel, Field

# find_code函數 通過國家的名稱來獲得ISO3166的編碼
def find_code(csv_file_path, country_name) -> str:
    """
    根據國家的名字 返回該區域的編碼
    :param csv_file_path: API提供的地區名與經度、緯度對換的CSV文件
    :param country_name: 國家名字
    :return:
    """
    # csv文件數據的解析
    country_map = {}
    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            iso3166_code = row["alpha-2"].strip()
            country = row["name"].strip()
            if country not in country_map:
                country_map[country] = iso3166_code

    return country_map.get(country_name, None)

# 數據模型 設置大模型要獲析的數據字段
class WeatherInputArgs(BaseModel):
    """
    Input的Schema類
    """
    location: str = Field(..., description="用於查詢全球各國天氣的位置信息")  # description是給ai看的 一定要寫準備
    # location: str = Field(..., description="用於查詢天氣的位置信息")

# 設置工具類 具體參數可參考LangChain中的TavilySearchResults
class WeatherTool(BaseTool):
    """查詢實時天氣的工具"""
    name: str = "weather_tool"  # 工具的名字
    description: str = "可以查詢任意位置的當前天氣情況"  # 工具的描述 需清晰寫明服務內容 這與大模型決策有關
    args_schema: Type[WeatherInputArgs] = WeatherInputArgs  # 工具要傳的參數 這里就只定義了location一個參數

    def _run(
            self,
            location: str,  # 前面定義了location是str 所以只能傳str
            run_manager: Optional[CallbackManagerForToolRun] = None,  # 這是固定的寫法
    ) -> str:
        """就是調用工具的時候，自動執行的函數"""
        iso3166_code = find_code("all.csv", location)  # 返回的國家iso3166編碼
        print(f"需要查詢的{location},的ISO3166代碼是:{iso3166_code}")

        # 獲取經度、緯度的API
        url_env = f"http://api.openweathermap.org/geo/1.0/direct?q={iso3166_code}&appid=8f2e5c5b421bf810d537bc0e12399fa1"

        # 發送請求
        response_env = requests.get(url_env)

        # 將返回的內容設為JSON格式
        data_env = response_env.json()  # 官方文檔規定設回值為json格式

        # 解析數據
        lat = data_env[0]["lat"]
        lon = data_env[0]["lon"]

        # 通過把經度、緯度的傳參加到API中 獲取對應地區的天氣數據
        url_temp = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid=8f2e5c5b421bf810d537bc0e12399fa1&lang=zh-tw"

        # 發送請求
        response_temp = requests.get(url_temp)

        # 將返回的內容設為JSON格式
        data_temp = response_temp.json()

        # 解析數據 由於返回的溫度數據單位為開氏 因而需要減273.15
        weather = data_temp["weather"][0]["description"]
        temp = data_temp["main"]["temp"]-273.15
        feels_like = data_temp["main"]["feels_like"]-273.15
        temp_min = data_temp["main"]["temp_min"]-273.15
        temp_max = data_temp["main"]["temp_max"]-273.15
        humidity = data_temp["main"]["humidity"]
        name = data_temp["name"]

        # 調用工具 返回的數據樣式
        return f"位置: {location} 當前天氣: {weather} 溫度: {temp}C 最小溫度: {temp_min} 最大溫度: {temp_max} 體感溫度: {feels_like} 相對濕度: {humidity}"

# 主程序
def main_loop():

    # 配置大模型gemini
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # 設置工具
    tools = [WeatherTool()]

    # 定義提示模板 (輸入給大模型的模板)
    prompt = ChatPromptTemplate.from_messages([
        # 大模型提示
        SystemMessage(content="你是一個智能天氣查詢工具。\n你將根據用戶的要求返回對應真實的天氣信息。"),
        # 用戶提示
        HumanMessage(content="{text}")
    ])

    # 創建代理
    agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools=tools)

    # 創建鏈
    chain = prompt | agent_executor

    # 返回鏈
    return chain


if __name__ == "__main__":

    # 執行main_loop函數
    chain = main_loop()

    # 使用FastAPI發佈服務
    app = FastAPI(title="我的Langchain服務", version="v1.0",
                  description="智能天氣查詢工具")  # 這步是創建了fastapi的應用

    # 添加路由
    add_routes(
        app,
        chain,  # 可以傳鏈/大模型
        path="/chain"  # 路由可以隨便改
    )

    # 運行伺服器
    uvicorn.run(app, host="localhost", port=8002)

