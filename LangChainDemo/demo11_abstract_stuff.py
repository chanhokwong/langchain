import os, datetime

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic.v1 import BaseModel, Field

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

# 設定版本為LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"  # 設置LangSmith的項目名稱 LangSmith是LangChain中的子項目 用以項目數據追蹤
os.environ["LANGCHAIN_API_KEY"] = "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"

# 聊天機器人案例
# 創建模型
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)  # temperature影響多樣性 越高越隨機

# 加載我們的文檔 我們將使用WebBaseLoader來加載博客文章
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()  # 得到整篇文章

# 第一種寫法 Stuff
# chain = load_summarize_chain(model, chain_type="stuff")

# 第二種寫法 Stuff
prompt_template = """針對下面的內容，寫一個簡潔的總結摘要:
"{text}"
簡潔的總結摘要:"""
prompt = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(llm=model, prompt=prompt)
# LLMChain是過時的 不過仍然可以使用 不替換的原因主要是因為還要用StuffDocumentsChain這個仍然新的類
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
# llm_chain = LLMChain(llm=model, prompt=prompt)

stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='text')

result = stuff_chain.invoke(docs)
print(result["output_text"])  # "output_text"這個是固定參數
# 內容應該不太準確 因為大模型或者只讀了前面文篇的一部分 超過token可能不夠準確 沒有超過就應該是最終版本

# 想要中文的話 可以加提示
