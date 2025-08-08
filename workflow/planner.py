from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from agents.intent_agent import classify_intent
from agents.task_decompose_agent import task_decompose

class State(TypedDict):
    """
    定义工作流中各个节点间传递的状态信息

    Attributes:
        user_input: 用户输入的指令(instruction)
        intent: 大模型识别的意图(intent)
        sub_tasks: 大模型根据识别的意图拆解的任务(tasks)
        pre_node_result: 上一个节点的结果
        all_result: 所有任务的结果
        current_task_index: 当前任务的索引
    """
    user_input: str
    intent: str
    sub_tasks: list[str]
    pre_node_result: str
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
    sub_tasks = task_decompose(state["intent"])
    return {
        **state,
        "sub_tasks": sub_tasks,
    }