# LangChain開發
具體內容主要分為兩部分，示例和項目



# LangChain示例

**示例1: LangChain調用大模型Gemini API**

概念: 使用Google內部的API調用方法


**示例2: LangChain大模型與對話歷史紀錄**

概念: 創建一個空字典，然後設置對話主題，接著把所有的對話都儲存到字典當中

**示例3: LangChain的嵌入與向量數據庫**

概念: 

**示例4: LangChain為大模型添加搜索工具**

概念: 

**示例5: LangChain檢索增強生成RAG 樣例**

概念: 

**示例6: LangChain與數據庫的互動**

概念:

**示例7: Langchain檢索Youtube視頻字幕**

概念:

**示例8: LangChain提取結構化數據**

概念:

**示例9: LangChain自動生成數據**

概念: 

**示例10: LangChain文本分類器**

概念:

**示例11: LangChain文本摘要-Stuff**

概念:

**示例12: LangChain文本摘要-Map Reduce**

概念: 

**示例13: LangChain文本摘要-Refine**

概念: 

# LangChain項目

**項目1: 智能地區天氣查詢助手 WeatherTools**

功能: 可以快速查詢各個地區的實時天氣數據

內容: 設置數據模型來讓大模型獲取國家字段信息，然後通過國家名稱對應ISO3316編碼的CSV文件來獲取ISO3316編碼(API規定)，接著再把獲得的ISO3316編碼傳至OpenWeather的API_KEY來獲取地區的經度、緯度，最後再調用API來獲取地區的天氣數據。

OpenWeather地理API的調用: https://openweathermap.org/api/geocoding-api

OpenWeather天氣API的調用: https://openweathermap.org/current

CSV數據集來源: https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv

項目缺點: 此API無法精準至每個區域，主要原因是因為沒有對應可用的CSV文件，如果有細至如香港每個區對應的經度、緯度的CSV開放數據集文件，那這個天氣查詢便可精確查詢至每個地方。




**項目2: 智能翻譯工具 TranslateTools**

功能: 可以快速翻譯各國語言的句子

內容: 通過在模板中設置大模型和用戶的傳參變量，以讓用戶在使用翻譯服務時，可以傳入要翻譯的句子和要翻譯的語言。
