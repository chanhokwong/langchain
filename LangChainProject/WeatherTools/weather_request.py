import requests, json

url = "http://127.0.0.1:8002/chain/invoke"

headers = {  # 這個是必須要設置的 否則會出現格式錯誤的情況
    "Content-Type": "application/json"
}

data = {
    "input":{
        "text":'你能說一下今天香港的天氣如何嗎?'
    }
}

json_data = json.dumps(data,ensure_ascii=False).encode('utf-8')

res = requests.post(url=url,headers=headers,data=json_data)

print(res.status_code)
print(res.json())
print("================")
print(res.json()["output"]["messages"][4]["content"])




