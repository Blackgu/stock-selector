from agents.model import get_tongyi_llm
from langchain_core.prompts import ChatPromptTemplate
from utils.prompt_generator import generate_prompt

llm = get_tongyi_llm()

system_prompt = """
你是一个选股票的意图分类助手，请将用户的请求分类为以下意图之一：
`stock_analysis`, `search_stock`，`unknow`。
按提供的示例返回JSON报文，里面包含意图标签和附加参数。
"""

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

examples = [
    {
        "input": "我要找出市值大于 100 亿美金的纳斯达克股票",
        "output": {
            "intent": "search_stock",
            "condition": "市值大于 100 亿美金的纳斯达克股票"
        }
    },
    {
        "input": "帮我分析下AAPL这支股票",
        "output": {
            "intent": "stock_analysis",
            "stock": "AAPL"
        }
    }
]

def classify_intent(user_input):
    """
    根据用户输入获取意图分类结果

    该函数使用预定义的提示模板和大语言模型对用户输入进行意图分类，
    将用户请求分类为预定义的意图标签之一。

    Args:
        user_input (str): 用户的输入文本

    Returns:
        str: 意图分类结果，可能的值包括:
            - 'stock_analysis': 股票分析意图
            - 'search_stock': 股票搜索意图
            - 'unknow': 未知意图
    """
    final_prompt = generate_prompt(system_prompt, example_prompt, examples)

    message = final_prompt.format_prompt(user_input=user_input).to_messages()
    intent_response =  llm.invoke(message)
    return intent_response.content
