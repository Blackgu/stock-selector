from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

def generate_prompt(system_prompt: str, example_prompt: str, examples: list[dict]):
    """
    根据系统提示、示例模板和示例列表生成最终的提示模板

    Args:
        system_prompt (str): 系统提示信息，用于定义模型的行为和角色
        example_prompt (str): 示例模板，定义了输入输出对的格式
        examples (list[dict]): 示例列表，每个示例包含具体的输入输出对

    Returns:
        ChatPromptTemplate: 组合后的提示模板，包含系统提示、示例和用户输入占位符
    """

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "{user_input}")
        ]
    )
    return final_prompt
