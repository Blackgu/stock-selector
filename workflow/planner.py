from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from agents.intent_agent import classify_intent
from agents.task_decompose_agent import decompose_task
from agents.task_execute_agent import execute_task

class State(TypedDict):
    """
    定义工作流中各个节点间传递的状态信息

    Attributes:
        user_input: 用户输入的指令(instruction)
        intent: 大模型识别的意图(intent)
        sub_tasks: 大模型根据识别的意图拆解的任务(tasks)
        previous_result: 上一个节点的结果
        all_result: 所有任务的结果
        current_task_index: 当前任务的索引
    """
    user_input: str
    intent: str
    sub_tasks: list[str]
    previous_result: str
    all_result: list[str]
    current_task_index: int


def intent_node(state: State) -> State:
    """
    处理用户输入并识别意图的节点函数

    Args:
        state (State): 包含用户输入和当前状态的字典

    Returns:
        State: 更新后的状态，包含识别出的意图
    """
    intent = classify_intent(state["user_input"])
    return {
        **state,
        "intent": intent,
    }

def decompose_node(state: State) -> State:
    """
    根据识别出的意图分解为具体子任务的节点函数

    Args:
        state (State): 包含已识别意图和当前状态的字典

    Returns:
        State: 更新后的状态，包含分解出的子任务列表
    """
    sub_tasks = decompose_task(state["intent"])
    return {
        **state,
        "sub_tasks": sub_tasks,
    }

def execute_node(state: State) -> State:
    sub_tasks = state["sub_tasks"]
    current_task_index = state["current_task_index"]
    if current_task_index >= len(sub_tasks):
        return state

    task = sub_tasks[current_task_index]
    prompt = f"现在执行任务：'{task.task}'，请生成输出。"
    if current_task_index > 0:
        previous_task = sub_tasks[current_task_index - 1]
        previous_result = state["all_result"][current_task_index - 1]
        prompt = f"根据上一个任务('{previous_task.task}')的结果：'{previous_result}'，现在执行任务：'{task.task}'，请生成输出。"

    result = execute_task()

    return {
        **state,
        "previous_result": result,
        "all_results": state.get("all_results", []) + [result],
        "current_task_index": current_task_index + 1
    }