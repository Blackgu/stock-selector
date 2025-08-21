from agents.model import get_tongyi_llm
from langchain_core.prompts import PromptTemplate

llm = get_tongyi_llm()

SYSTEM_PROMPT = (
    "你是一个任务执行助手。\n"
    "你可以：\n"
    "1) 调用工具（function calling、tool、MCP），例如 web_search；\n"
    "2) 编写 Python 代码，并放在一个单独的 ```python 代码块``` 中，并在 Docker 沙箱中执行；\n"
    "3) 当需要多个步骤时，可以先获取数据（用工具或代码），再综合分析并给出最终结论。\n"
    "注意：\n"
    "- 如果需要执行代码，请务必只输出一个纯粹的 python 代码块（不写注释）；\n"
    "- 如果要调用工具，请使用 function call、tool、MCP；\n"
    "- 输出最终结论时，用清晰的小结。\n"
)

first_execute_task_prompt = PromptTemplate(
    input_variables=["task_content"],
    template="现在执行任务：{task_content}，请执行并生成输出。"
)

execute_task_prompt = PromptTemplate(
    input_variables=["previous_task_content", "previous_result", "task_content"],
    template="根据上一个任务('{previous_task_content}')的结果：'{previous_result}'，现在执行任务：'{task_content}'，请执行并生成输出。"
)

def execute_task(task: str, previous_task: str, previous_result: str) -> dict:
    """
    执行给定的任务，根据上一个任务的结果进行上下文相关的处理

    Args:
        task (str): 当前需要执行的任务内容
        previous_task (str): 上一个任务的内容，如果为初始任务则为None
        previous_result (str): 上一个任务的执行结果，如果为初始任务则为None

    Returns:
        dict: 包含任务名和执行结果的字典，格式为 {"task": task, "result": result_content}
    """
    # 构造初始任务提示词
    prompt = first_execute_task_prompt.format(task_content=task)
    # 如果存在上一个任务和结果，则使用上下文相关的提示模板
    if previous_task is not None and previous_result is not None:
        prompt = execute_task_prompt.format(
            previous_task_content=previous_task,
            previous_result=previous_result,
            task_content=task)

    task_execute_response = llm.invoke(prompt)
    return {"task": task, "result": task_execute_response.content}