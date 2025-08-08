from agents.model import get_tongyi_llm
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from utils.prompt_generator import generate_prompt

llm = get_tongyi_llm()

system_prompt = """
你是一个任务规划助手，负责根据传入的用户意图，分解为可以执行的子任务。
这些子任务可以各自独立执行，也可以按一定的顺序依次执行

你的目标是：
1. 明确每个子任务的目的
2. 保持子任务之间的执行逻辑清晰（先后顺序）
3. 每个子任务一行文字描述，简洁有力

返回格式按示例要求为JSON报文，里面包含一个List，List里包含拆解的子任务，每个子任务内容为文字描述
"""

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

examples = [
    {
        "input": {
            "intent": "search_info",
            "condition": "市值大于 100 亿美金的纳斯达克股票"
        },
        "output": {
            [
                "获取所有在纳斯达克交易所上市的股票列表",
                "为每支股票获取其最新的市值信息",
                "筛选出市值大于 100 亿美金（100B）的股票",
                "整理结果，输出为包含股票代码、名称、市值的列表"
            ]
        }
    },
    {
        "input": {
            "intent": "stock_analysis",
            "stock": "AAPL"
        },
        "output": [
            "获取 AAPL 股票的最新价格和历史走势数据（如近6个月）",
            "获取 AAPL 最新财报摘要（营收、净利润、EPS、现金流等）",
            "计算并分析关键财务指标（市盈率、净利率、同比增长等）",
            "获取 AAPL 所在行业和市场的整体表现做对比分析",
            "总结 AAPL 股票的当前估值、业绩表现和市场位置"
        ]
    }
]

def task_decompose(intent):
    """
    根据用户意图分解为具体的子任务列表

    Args:
        intent (str): 用户的意图描述，用于生成具体的子任务

    Returns:
        str: 任务分解结果，包含子任务列表的字符串表示
    """
    final_prompt = generate_prompt(system_prompt, example_prompt, examples)

    task_decompose_message = final_prompt.format_prompt(user_input=intent).to_messages()
    task_decompose_response = llm.invoke(task_decompose_message)
    return task_decompose_response.content
