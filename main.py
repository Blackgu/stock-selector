from settings import logger
from agents.intent_agent import classify_intent
from agents.task_decompose_agent import decompose_task

if __name__ == '__main__':
    intent = classify_intent("TSLA这支股票怎么样？")
    logger.info(intent)
    sub_tasks_response = decompose_task(intent)
    logger.info(sub_tasks_response)
