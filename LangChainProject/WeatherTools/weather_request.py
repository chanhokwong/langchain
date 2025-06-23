import requests, json

# 根據FastAPI的設置來寫 需要注意在最後加上/invoke調用服務
url = "http://127.0.0.1:8002/chain/invoke"

headers = {  # 這個是必須要設置的 否則會出現格式錯誤的情況
    "Content-Type": "application/json"  # json格式
}

data = {
    "input":{
        "text":'你能說一下今天香港的天氣如何嗎?'  # text對應服務中的輸入變量
    }
}

json_data = json.dumps(data,ensure_ascii=False).encode('utf-8')  # 因應中文編碼而設置

# 發送請求
res = requests.post(url=url,headers=headers,data=json_data) 

# 打印請求狀態碼 成功:200 錯誤:404,500,... 
print(res.status_code)
# 打印json格式的返回內容
print(res.json())
# 打印AI具體返回的內容
print(res.json()["output"]["messages"][4]["content"])




