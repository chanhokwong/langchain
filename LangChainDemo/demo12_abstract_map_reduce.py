import os, datetime

from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import CharacterTextSplitter
from pydantic.v1 import BaseModel, Field

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

# 設定版本為LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"  # 設置LangSmith的項目名稱 LangSmith是LangChain中的子項目 用以項目數據追蹤
os.environ["LANGCHAIN_API_KEY"] = "your_tracing_api_key"

# 聊天機器人案例
# 創建模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)  # temperature影響多樣性 越高越隨機

# 加載我們的文檔 我們將使用WebBaseLoader來加載博客文章
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()  # 得到整篇文章

# 第二種 Map-Reduce
# 第一步: 切割階段
# 每一個小docs為1000個token
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)  # chunk_size是每個文本的token上限
split_docs = text_splitter.split_documents(docs)

# 第二步: map階段
map_template = """
以下是一組文檔(documents)
"{docs}"
根據這個文檔列表，請給出總結摘要:
"""
map_prompt = PromptTemplate.from_template(map_template)
map_llm_chain = LLMChain(llm=model, prompt=map_prompt)

# 第三步: reduce階段: (combine 和 最終的reduce)
reduce_template = """
以下是一組總結摘要:
{docs}
將這些內容提煉成一個最終的、統一的總結摘要:
"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_llm_chain = LLMChain(llm=model, prompt=reduce_prompt)

"""
reduce的思路: 
如果map之後文檔的累積token數超過了 4000個，那麼我們將递歸地將文檔以<=4000 個token的批次傳遞給我們的StuffDocumentChain做一次匯總
一旦這些批量摘要的累積大小小於4000個token 我們將它們全部傳递給StuffDocumentsChain 最後一次 以創建最終摘要
"""

# 定義一個combine的chain
combine_chain = StuffDocumentsChain(llm_chain=reduce_llm_chain, document_variable_name='docs')

reduce_chain = ReduceDocumentsChain(
    # 這是最終調用的chain
    combine_documents_chain=combine_chain,
    # 中間的匯總的臉
    collapse_documents_chain=combine_chain,
    # 將文檔分組的最大令牌數
    token_max=4000
)

# 第四步: 合併所有的鏈
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_llm_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name="docs",
    return_intermediate_steps=False  # return_intermediate_steps是指要不要返回中間還未匯總完的數據
)

# 第五步: 調用最終的鏈
result = map_reduce_chain.invoke(split_docs)

print(result["output_text"])  # "output_text"這個是固定參數
