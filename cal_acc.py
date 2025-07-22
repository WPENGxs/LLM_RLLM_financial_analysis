from tqdm import tqdm
import json
from tabulate import tabulate

def truncate(number, decimal_places):
    factor = 10 ** decimal_places
    return int(number * factor) / factor

def cal_acc(input_list):
    count_of_ones = input_list.count(1)
    total = len(input_list)
    if total == 0:
        return '-'
    acc = count_of_ones / total
    acc = acc * 100
    acc = truncate(acc, 2)
    return acc

def cal_prompting_acc(json_path):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        log_data = json.load(json_file)

    arithmetic_correct_list = []
    arithmetic_correct_list_easy = []
    arithmetic_correct_list_medium = []
    arithmetic_correct_list_hard = []

    non_arithmetic_correct_list = []
    non_arithmetic_correct_list_easy = []
    non_arithmetic_correct_list_medium = []
    non_arithmetic_correct_list_hard = []

    overall_correct_list = []

    correct_en = []
    correct_zh = []
    correct_fr = []

    print(f'processing: {json_path}')
    with tqdm(total=len(log_data)) as pbar:
        for d in log_data:
            idx = d['idx']
            answers = d['answers']
            topic_difficulty = d['topic_difficulty']
            question_type = d['question_type']
            language = d['language']
            is_arithmetic = d['is_arithmetic']
            predict_answers = d['predict_answers']
            correct = d['correct']

            if is_arithmetic == 1:
                arithmetic_correct_list.append(correct)
                if topic_difficulty == 'easy':
                    arithmetic_correct_list_easy.append(correct)
                elif topic_difficulty == 'medium':
                    arithmetic_correct_list_medium.append(correct)
                elif topic_difficulty == 'hard':
                    arithmetic_correct_list_hard.append(correct)
            elif is_arithmetic == 0:
                non_arithmetic_correct_list.append(correct)
                if topic_difficulty == 'easy':
                    non_arithmetic_correct_list_easy.append(correct)
                elif topic_difficulty == 'medium':
                    non_arithmetic_correct_list_medium.append(correct)
                elif topic_difficulty == 'hard':
                    non_arithmetic_correct_list_hard.append(correct)
            if language == 'English':
                correct_en.append(correct)
            elif language == 'Chinese':
                correct_zh.append(correct)
            elif language == 'French':
                correct_fr.append(correct)
            overall_correct_list.append(correct)

            pbar.update(1)
    data = [
        ["Arithmetic", "-", "-", "-", "Non-Arithmetic", "-", "-", "-", "Overall"],
        ["Overall", "Easy", "Medium", "Hard", "Overall", "Easy", "Medium", "Hard", "Overall"],
        [cal_acc(arithmetic_correct_list), cal_acc(arithmetic_correct_list_easy), cal_acc(arithmetic_correct_list_medium), cal_acc(arithmetic_correct_list_hard), cal_acc(non_arithmetic_correct_list), cal_acc(non_arithmetic_correct_list_easy), cal_acc(non_arithmetic_correct_list_medium), cal_acc(non_arithmetic_correct_list_hard), cal_acc(overall_correct_list)]
    ]
    print(tabulate(data, headers="firstrow", tablefmt="grid"))
    print(f'Overall Acc:\nEn: {cal_acc(correct_en)}, Zh: {cal_acc(correct_zh)}, Fr: {cal_acc(correct_fr)}, Zh and Fr: {cal_acc(correct_zh+correct_fr)}')
    # print(f'Overall Acc:\nEasy: {cal_acc(arithmetic_correct_list_easy+non_arithmetic_correct_list_easy)}, Medium: {cal_acc(arithmetic_correct_list_medium+non_arithmetic_correct_list_medium)}, Hard: {cal_acc(arithmetic_correct_list_hard+non_arithmetic_correct_list_hard)}')

def cal_agent_acc(json_path):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        log_data = json.load(json_file)

    arithmetic_correct_list_self_refine = []
    arithmetic_correct_list_easy_self_refine = []
    arithmetic_correct_list_medium_self_refine = []
    arithmetic_correct_list_hard_self_refine = []
    non_arithmetic_correct_list_self_refine = []
    non_arithmetic_correct_list_easy_self_refine = []
    non_arithmetic_correct_list_medium_self_refine = []
    non_arithmetic_correct_list_hard_self_refine = []
    overall_correct_list_self_refine = []

    arithmetic_correct_list_s3_agent = []
    arithmetic_correct_list_easy_s3_agent = []
    arithmetic_correct_list_medium_s3_agent = []
    arithmetic_correct_list_hard_s3_agent = []
    non_arithmetic_correct_list_s3_agent = []
    non_arithmetic_correct_list_easy_s3_agent = []
    non_arithmetic_correct_list_medium_s3_agent = []
    non_arithmetic_correct_list_hard_s3_agent = []
    overall_correct_list_s3_agent = []

    print(f'processing: {json_path}')
    with tqdm(total=len(log_data)) as pbar:
        for d in log_data:
            idx = d['idx']
            answers = d['answers']
            topic_difficulty = d['topic_difficulty']
            question_type = d['question_type']
            language = d['language']
            is_arithmetic = d['is_arithmetic']

            predict_answers_self_refine = d['predict_answers_self_refine']
            correct_self_refine = d['correct_self_refine']
            predict_answers_s3_agent = d['predict_answers_s3_agent']
            correct_s3_agent = d['correct_s3_agent']

            if is_arithmetic == 1:
                arithmetic_correct_list_self_refine.append(correct_self_refine)
                arithmetic_correct_list_s3_agent.append(correct_s3_agent)
                if topic_difficulty == 'easy':
                    arithmetic_correct_list_easy_self_refine.append(correct_self_refine)
                    arithmetic_correct_list_easy_s3_agent.append(correct_s3_agent)
                elif topic_difficulty == 'medium':
                    arithmetic_correct_list_medium_self_refine.append(correct_self_refine)
                    arithmetic_correct_list_medium_s3_agent.append(correct_s3_agent)
                elif topic_difficulty == 'hard':
                    arithmetic_correct_list_hard_self_refine.append(correct_self_refine)
                    arithmetic_correct_list_hard_s3_agent.append(correct_s3_agent)
            elif is_arithmetic == 0:
                non_arithmetic_correct_list_self_refine.append(correct_self_refine)
                non_arithmetic_correct_list_s3_agent.append(correct_s3_agent)
                if topic_difficulty == 'easy':
                    non_arithmetic_correct_list_easy_self_refine.append(correct_self_refine)
                    non_arithmetic_correct_list_easy_s3_agent.append(correct_s3_agent)
                elif topic_difficulty == 'medium':
                    non_arithmetic_correct_list_medium_self_refine.append(correct_self_refine)
                    non_arithmetic_correct_list_medium_s3_agent.append(correct_s3_agent)
                elif topic_difficulty == 'hard':
                    non_arithmetic_correct_list_hard_self_refine.append(correct_self_refine)
                    non_arithmetic_correct_list_hard_s3_agent.append(correct_s3_agent)
            overall_correct_list_self_refine.append(correct_self_refine)
            overall_correct_list_s3_agent.append(correct_s3_agent)

            pbar.update(1)
    data = [
        ["method", "Arithmetic", "-", "-", "-", "Non-Arithmetic", "-", "-", "-", "Overall"],
        ["-", "Overall", "Easy", "Medium", "Hard", "Overall", "Easy", "Medium", "Hard", "Overall"],
        ["Self-Refine", cal_acc(arithmetic_correct_list_self_refine), cal_acc(arithmetic_correct_list_easy_self_refine), cal_acc(arithmetic_correct_list_medium_self_refine), cal_acc(arithmetic_correct_list_hard_self_refine), cal_acc(non_arithmetic_correct_list_self_refine), cal_acc(non_arithmetic_correct_list_easy_self_refine), cal_acc(non_arithmetic_correct_list_medium_self_refine), cal_acc(non_arithmetic_correct_list_hard_self_refine), cal_acc(overall_correct_list_self_refine)],
        ["S3 Agent", cal_acc(arithmetic_correct_list_s3_agent), cal_acc(arithmetic_correct_list_easy_s3_agent), cal_acc(arithmetic_correct_list_medium_s3_agent), cal_acc(arithmetic_correct_list_hard_s3_agent), cal_acc(non_arithmetic_correct_list_s3_agent), cal_acc(non_arithmetic_correct_list_easy_s3_agent), cal_acc(non_arithmetic_correct_list_medium_s3_agent), cal_acc(non_arithmetic_correct_list_hard_s3_agent), cal_acc(overall_correct_list_s3_agent)]
    ]
    print(tabulate(data, headers="firstrow", tablefmt="grid"))

def cal_multilingual_acc(json_path):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        log_data = json.load(json_file)

    en_en_correct_list = []
    en_zh_correct_list = []
    en_fr_correct_list = []

    zh_en_correct_list = []
    zh_zh_correct_list = []
    zh_fr_correct_list = []

    fr_en_correct_list = []
    fr_zh_correct_list = []
    fr_fr_correct_list = []

    print(f'processing: {json_path}')
    with tqdm(total=len(log_data)) as pbar:
        for d in log_data:
            idx = d['idx']
            answers = d['answers']
            topic_difficulty = d['topic_difficulty']
            question_type = d['question_type']
            language = d['language']
            is_arithmetic = d['is_arithmetic']

            if language == 'English':
                correct_clp_en = d['correct_clp_en']
                correct_clp_zh = d['correct_clp_zh']
                correct_clp_fr = d['correct_clp_fr']
                en_en_correct_list.append(correct_clp_en)
                en_zh_correct_list.append(correct_clp_zh)
                en_fr_correct_list.append(correct_clp_fr)
            elif language == 'Chinese':
                correct_clp_en = d['correct_clp_en']
                correct_clp_zh = d['correct_clp_zh']
                correct_clp_fr = d['correct_clp_fr']
                zh_en_correct_list.append(correct_clp_en)
                zh_zh_correct_list.append(correct_clp_zh)
                zh_fr_correct_list.append(correct_clp_fr)
            elif language == 'French':
                correct_clp_en = d['correct_clp_en']
                correct_clp_zh = d['correct_clp_zh']
                correct_clp_fr = d['correct_clp_fr']
                fr_en_correct_list.append(correct_clp_en)
                fr_zh_correct_list.append(correct_clp_zh)
                fr_fr_correct_list.append(correct_clp_fr)

            pbar.update(1)
    data = [
        ["en", "-", "zh", "-", "fr", "-",],
        ["zh", "fr", "en", "fr", "en", "zh"],
        [cal_acc(en_zh_correct_list), cal_acc(en_fr_correct_list), cal_acc(zh_en_correct_list), cal_acc(zh_fr_correct_list), cal_acc(fr_en_correct_list), cal_acc(fr_zh_correct_list)]
    ]
    # data = [
    #     ["en", "-", "-", "zh", "-", "-", "fr", "-", "-"],
    #     ["en", "zh", "fr", "en", "zh", "fr", "en", "zh", "fr"],
    #     [cal_acc(en_en_correct_list), cal_acc(en_zh_correct_list), cal_acc(en_fr_correct_list), cal_acc(zh_en_correct_list), cal_acc(zh_zh_correct_list), cal_acc(zh_fr_correct_list), cal_acc(fr_en_correct_list), cal_acc(fr_zh_correct_list), cal_acc(fr_fr_correct_list)]
    # ]
    print(tabulate(data, headers="firstrow", tablefmt="grid"))
    print(f'Overall Acc (CLP: Zh-> En): {cal_acc(zh_en_correct_list)}')
    print(f'Overall Acc (CLP: Fr-> En): {cal_acc(fr_en_correct_list)}')
    print(f'Overall Acc (CLP: Zh, Fr -> En): {cal_acc(zh_en_correct_list+fr_en_correct_list)}')

if __name__ == '__main__':
    model_list = ['Meta-Llama-3.1-8B-Instruct', 'gpt-4o-mini', 'gemini-1.5-flash', 'qwen2.5-32b-instruct', 'deepseek-chat', 'DeepSeek-R1-Distill-Qwen-32B', 'Qwen3-14B', 'o4-mini']
    # model_list = ['gpt-4o-mini']

    for model in model_list:
        cal_prompting_acc(f'./log/eval/prompting/{model}-direct.json')
        cal_prompting_acc(f'./log/eval/prompting/{model}-cot.json')
        cal_prompting_acc(f'./log/eval/prompting/{model}-ps.json')
        cal_agent_acc(f'./log/eval/agent/{model}.json')
        cal_prompting_acc(f'./log/eval/translate/{model}.json')
        cal_multilingual_acc(f'./log/eval/multilingual/{model}.json')
    
    # cal_prompting_acc(f'./log/eval/prompting/qwen3-14b-direct-non-thinking.json')
    # cal_prompting_acc(f'./log/eval/prompting/qwen3-14b-cot-non-thinking.json')
    # cal_prompting_acc(f'./log/eval/prompting/qwen3-14b-ps-non-thinking.json')
    # cal_agent_acc(f'./log/eval/agent/qwen3-14b-non-thinking.json')