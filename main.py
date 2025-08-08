from settings import logger
from agents.intent_agent import classify_intent
from agents.task_decompose_agent import task_decompose

if __name__ == '__main__':
    intent_response = classify_intent("TSLA这支股票怎么样？")
    logger.info(intent_response.content)
    sub_tasks_response = task_decompose(intent_response.content)
    logger.info(sub_tasks_response.content)
