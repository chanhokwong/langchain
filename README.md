# LangChain開發
**項目1: 智能地區天氣查詢助手**

功能: 可以快速查詢各個地區的實時天氣數據

內容: 設置數據模型來讓大模型獲取國家字段信息，然後通過國家名稱對應ISO3316編碼的CSV文件來獲取ISO3316編碼(API規定)，接著再把獲得的ISO3316編碼傳至OpenWeather的API_KEY來獲取地區的經度、緯度，最後再調用API來獲取地區的天氣數據。

OpenWeather地理API的調用: https://openweathermap.org/api/geocoding-api

OpenWeather天氣API的調用: https://openweathermap.org/current

CSV數據集來源: https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv

項目缺點: 此API無法精準至每個區域，主要原因是因為沒有對應可用的CSV文件，如果有細至如香港每個區對應的經度、緯度的CSV開放數據集文件，那這個天氣查詢便可精確查詢至每個地方。
