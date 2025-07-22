
def get_base_prompt(input, source_language, question_type):
    if question_type == 'open question':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the question. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```. Your answer needs to use {source_language}.\n\n{input}'
    elif question_type == 'multiple-choice':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the question. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "Your options, separated by commas, such as A, B, C"}}\n```.\n\n{input}'
    return prompt

def get_CoT_prompt(input, source_language, question_type):
    # https://arxiv.org/abs/2205.11916
    # Let\'s think step by step!
    if question_type == 'open question':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the question. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```. Your answer needs to use {source_language}.\n\n{input}\n\nLet\'s think step by step!'
    elif question_type == 'multiple-choice':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the question. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "Your options, separated by commas, such as A, B, C"}}\n```.\n\n{input}\n\nLet\'s think step by step!'
    return prompt

def get_ps_prompt(input, source_language, question_type):
    # https://arxiv.org/abs/2305.04091
    # Let\'s first understand the problem and devise a plan to solve the problem. Then, let\'s carry out the plan to solve the problem step by step.
    if question_type == 'open question':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the question. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```. Your answer needs to use {source_language}.\n\n{input}\n\nLet\'s first understand the problem and devise a plan to solve the problem. Then, let\'s carry out the plan to solve the problem step by step.'
    elif question_type == 'multiple-choice':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the question. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "Your options, separated by commas, such as A, B, C"}}\n```.\n\n{input}\n\nLet\'s first understand the problem and devise a plan to solve the problem. Then, let\'s carry out the plan to solve the problem step by step.'
    return prompt

def get_self_refine_answer_prompt(input, source_language, question_type):
    # https://arxiv.org/abs/2303.17651
    if question_type == 'open question':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the question. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```. Your answer needs to use {source_language}.\n\n{input}\n\nLet\'s think step by step!'
    elif question_type == 'multiple-choice':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the question. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "Your options, separated by commas, such as A, B, C"}}\n```.\n\n{input}\n\nLet\'s think step by step!'
    return prompt

def get_self_refine_feedback_prompt(input):
    prompt = f'Please judge whether the answer is correct based on the context and the corresponding question. If it is correct, no improvement is needed. If it is incorrect, please indicate what needs to be modified.\n\n{input}'
    return prompt

def get_self_refine_refine_prompt(input, source_language, question_type):
    if question_type == 'open question':
        prompt = f'Please give a revised answer based on the current answer and suggestions. If no revision is required, please give the original answer. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```. Your answer needs to use {source_language}.\n\n{input}\n\nLet\'s think step by step!'
    elif question_type == 'multiple-choice':
        prompt = f'Please give a revised answer based on the current answer and suggestions. If no revision is required, please give the original answer. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "Your options, separated by commas, such as A, B, C"}}\n```.\n\n{input}\n\nLet\'s think step by step!'
    return prompt

def get_s3agent_1_prompt(input, source_language, question_type):
    # https://dl.acm.org/doi/abs/10.1145/3690642
    if question_type == 'open question':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the superficial expression. This includes analysis answer in contexts through context and question. Without considering conclusions drawn solely from context or question, both must be considered together. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```. Your answer needs to use {source_language}.\n\n{input}'
    elif question_type == 'multiple-choice':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the superficial expression. This includes analysis answer in contexts through context and question. Without considering conclusions drawn solely from context or question, both must be considered together. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "Your options, separated by commas, such as A, B, C"}}\n```.\n\n{input}'
    return prompt

def get_s3agent_2_prompt(input, source_language, question_type):
    if question_type == 'open question':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the semantic information. This includes analysis answer through context and question semantic. Without considering conclusions drawn solely from image context or question, both must be considered together. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```. Your answer needs to use {source_language}.\n\n{input}'
    elif question_type == 'multiple-choice':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the semantic information. This includes analysis answer through context and question semantic. Without considering conclusions drawn solely from image context or question, both must be considered together. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "Your options, separated by commas, such as A, B, C"}}\n```.\n\n{input}'
    return prompt

def get_s3agent_3_prompt(input, source_language, question_type):
    if question_type == 'open question':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the sentiment expression. This includes analysis answer on specific subjects in the image context and question. Without considering conclusions drawn solely from image context or question, both must be considered together. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```. Your answer needs to use {source_language}.\n\n{input}'
    elif question_type == 'multiple-choice':
        prompt = f'Now enter the context and the corresponding question, and please output the reasoning and correct answer based on the sentiment expression. This includes analysis answer on specific subjects in the image context and question. Without considering conclusions drawn solely from image context or question, both must be considered together. You should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "Your options, separated by commas, such as A, B, C"}}\n```.\n\n{input}'
    return prompt

def get_s3agent_final_prompt(input, source_language, question_type):
    if question_type == 'open question':
        prompt = f'Given three analysis, you need to analyze it from three perspectives: Superficial expression, Semantic information, and Sentiment expression. Combine these perspectives to output the reasoning and correct answer.\nFollow these rules:\n1. If any perspective cannot determine answer due to lack of information, disregard that perspective.\n2. If after one answer is disregard the remaining two views conflict, choose the answer with the most well-founded reasoning.\n3. Finally, you should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```. Your answer needs to use {source_language}.\n\n{input}'
    elif question_type == 'multiple-choice':
        prompt = f'Given three analysis, you need to analyze it from three perspectives: Superficial expression, Semantic information, and Sentiment expression. Combine these perspectives to output the reasoning and correct answer.\nFollow these rules:\n1. If any perspective cannot determine answer due to lack of information, disregard that perspective.\n2. If after one answer is disregard the remaining two views conflict, choose the answer with the most well-founded reasoning.\n3. Finally, you should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```.\n\n{input}'
    return prompt

def get_clp_stage1_prompt(input, clp_language, source_language):
    # https://arxiv.org/abs/2310.14799
    prompt = f'Please act as an expert in multi-lingual understanding in {source_language}.\n\nRequest:\n{input}\n\nLet\'s understand the task in {clp_language} step-by-step!'
    return prompt

def get_clp_stage2_prompt(clp_language, source_language, question_type):
    prompt = f'After understanding, you should act as an expert in arithmetic reasoning in {clp_language}.\nLet\'s resolve the task you understand above step-by-step!\n'
    if question_type == 'open question':
        prompt += f'Finally, you should format your answer as JSON format:\n```json\n{{"reasoning": "your reasoning", "answer": "your answer"}}\n```. Your answer needs to use {source_language}.'
    elif question_type == 'multiple-choice':
        prompt += 'Finally, you should format your answer as JSON format:\n```json\n{"reasoning": "your reasoning", "answer": "Your options, separated by commas, such as A, B, C"}\n```.'
    return prompt