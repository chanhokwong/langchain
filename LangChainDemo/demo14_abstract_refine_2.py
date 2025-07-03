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

prompt_template = """
針對下面的內容，寫一個簡潔的總結摘要:
"{text}"
簡潔的總結摘要:
"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary \n"
    "We have provided an existing summary up to certain point: {existing_answer}"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below. \n"
    "-------------\n"
    "{text}\n"
    "-------------\n"
    "\n"
    "Given the new context, refine the original summary in Chinese"
    "If the context isn't useful, return the original summary."
)

refine_prompt = PromptTemplate.from_template(refine_template)

chain = load_summarize_chain(
    llm=model,
    chain_type="refine",
    question_prompt=refine_prompt,
    return_intermediate_steps=False,
    input_key="input_documents",
    output_key="output_text",
)

# 第二種 Map-Reduce
# 第一步: 切割階段
# 每一個小docs為1000個token
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)  # chunk_size是每個文本的token上限
split_docs = text_splitter.split_documents(docs)

# 第五步: 調用最終的鏈
result = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)

print(result["output_text"])  # "output_text"這個是固定參數
