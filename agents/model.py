import os

from langchain_community.chat_models import ChatTongyi

TONGYI_API_KEY = os.getenv("DASHSCOPE_API_KEY")

tongyi_llm = ChatTongyi(
    api_key=TONGYI_API_KEY,
    model="qwen-plus-2025-07-14",
    top_p=1)

def get_tongyi_llm():
    return tongyi_llm