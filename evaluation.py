import argparse
import os
from tqdm import tqdm
from model import model
import json
import re
from cal_acc import *

parser = argparse.ArgumentParser()

parser.add_argument('--eval', type=str, default='prompting')
parser.add_argument('--log_path', type=str, default='./log')

def check_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"An error occurred while creating path '{path}': {e.strerror}")

def extract_options(text):
    if text == None:
        return []
    pattern = r'"answer":\s*"(.*?)"'
    match = re.search(pattern, text)
    
    if match:
        options = match.group(1)
        letters = re.findall(r'\b([A-Z])\b', options)
        return letters
    return []

def extract_response(text):
    pattern = r'"correct": (\d)'
    match = re.search(pattern, text)
    if match:
        correct = match.group(1)
        correct = int(correct)
        return correct
    else:
        return 0

def eval_prompting(log_path, model_list, method_list):
    eval_model = model('gpt-4o-mini')
    generator = eval_model.gpt_generator

    for m in model_list:
        for method in method_list:
            log = f'{log_path}/prompting/{m}-{method}.json'
            with open(log, 'r', encoding='utf-8') as json_file:
                log_data = json.load(json_file)

            eval_logs_path = f'{log_path}/eval/prompting/{m}-{method}.json'
            if os.path.exists(eval_logs_path):
                print(f'eval logs file already exists, will be skipped: {eval_logs_path}')
            else:
                eval_logs = []
                print(f'processing: {log}')
                with tqdm(total=len(log_data)) as pbar:
                    for d in log_data:
                        idx = d['idx']
                        answers = d['answers']
                        topic_difficulty = d['topic_difficulty']
                        question_type = d['question_type']
                        language = d['language']
                        is_arithmetic = d['is_arithmetic']

                        predict_answers = d['history']['answer']
                        correct = 0

                        if question_type == 'multiple-choice':
                            if answers == 'c.':
                                answers = ['C']
                            elif answers == 'b.':
                                answers = ['B']
                            else:
                                answers = answers.split(', ')
                            predict_answers = extract_options(predict_answers)
                            is_equal = set(answers) == set(predict_answers)
                            if is_equal:
                                correct = 1
                            else:
                                correct = 0
                        elif question_type == 'open question':
                            if predict_answers == None:
                                correct = 0
                            else:
                                prompt = f'''Please evaluate whether the predicted answer is correct in comparison to the standard answer. If it is correct, output {{"correct": 1}}; if it is incorrect, output {{"correct": 0}}.\nPredicted answer: {predict_answers}\nStandard answer: {answers}'''
                                response, history = generator(prompt)
                                correct = extract_response(response)
                        eval_log = {
                            "idx": idx,
                            "answers": answers,
                            "topic_difficulty": topic_difficulty,
                            "question_type": question_type,
                            "language": language,
                            "is_arithmetic": is_arithmetic,
                            "predict_answers": predict_answers,
                            "correct": correct
                        }
                        eval_logs.append(eval_log)
                        pbar.update(1)
                        # break
                    eval_logs_ = json.dumps(eval_logs, ensure_ascii=False)
                    eval_logs_file = open(eval_logs_path, 'w', encoding='utf-8')
                    eval_logs_file.write(eval_logs_)
                    eval_logs_file.close()


def eval_agent(log_path, model_list):
    eval_model = model('gpt-4o-mini')
    generator = eval_model.gpt_generator
    for m in model_list:
        log = f'{log_path}/agent/{m}.json'
        with open(log, 'r', encoding='utf-8') as json_file:
            log_data = json.load(json_file)

        eval_logs_path = f'{log_path}/eval/agent/{m}.json'
        if os.path.exists(eval_logs_path):
            print(f'eval logs file already exists, will be skipped: {eval_logs_path}')
        else:
            eval_logs = []
            print(f'processing: {log}')
            with tqdm(total=len(log_data)) as pbar:
                for d in log_data:
                    idx = d['idx']
                    answers = d['answers']
                    topic_difficulty = d['topic_difficulty']
                    question_type = d['question_type']
                    language = d['language']
                    is_arithmetic = d['is_arithmetic']

                    predict_answers_self_refine = d['self_refine_output']['output_refine']
                    correct_self_refine = 0
                    predict_answers_s3_agent = d['s3_agent_output']['s3_agent_final']
                    correct_s3_agent = 0

                    if question_type == 'multiple-choice':
                        if answers == 'c.':
                            answers = ['C']
                        elif answers == 'b.':
                            answers = ['B']
                        else:
                            answers = answers.split(', ')
                        predict_answers_self_refine = extract_options(predict_answers_self_refine)
                        is_equal = set(answers) == set(predict_answers_self_refine)
                        if is_equal:
                            correct_self_refine = 1
                        else:
                            correct_self_refine = 0
                        predict_answers_s3_agent = extract_options(predict_answers_s3_agent)
                        is_equal = set(answers) == set(predict_answers_s3_agent)
                        if is_equal:
                            correct_s3_agent = 1
                        else:
                            correct_s3_agent = 0
                    elif question_type == 'open question':
                        if predict_answers_self_refine == None:
                                correct_self_refine = 0
                        else:
                            prompt_self_refine = f'''Please evaluate whether the predicted answer is correct in comparison to the standard answer. If it is correct, output {{"correct": 1}}; if it is incorrect, output {{"correct": 0}}.\nPredicted answer: {predict_answers_self_refine}\nStandard answer: {answers}'''
                            response_self_refine, history_self_refine = generator(prompt_self_refine)
                            correct_self_refine = extract_response(response_self_refine)
                        if predict_answers_s3_agent == None:
                                correct_s3_agent = 0
                        else:
                            prompt_s3_agent = f'''Please evaluate whether the predicted answer is correct in comparison to the standard answer. If it is correct, output {{"correct": 1}}; if it is incorrect, output {{"correct": 0}}.\nPredicted answer: {predict_answers_s3_agent}\nStandard answer: {answers}'''
                            response_s3_agent, history_s3_agent = generator(prompt_s3_agent)
                            correct_s3_agent = extract_response(response_s3_agent)
                    eval_log = {
                        "idx": idx,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "predict_answers_self_refine": predict_answers_self_refine,
                        "correct_self_refine": correct_self_refine,
                        "predict_answers_s3_agent": predict_answers_s3_agent,
                        "correct_s3_agent": correct_s3_agent
                    }
                    eval_logs.append(eval_log)
                    pbar.update(1)
                    # break
                eval_logs_ = json.dumps(eval_logs, ensure_ascii=False)
                eval_logs_file = open(eval_logs_path, 'w', encoding='utf-8')
                eval_logs_file.write(eval_logs_)
                eval_logs_file.close()


def eval_multilingual(log_path, model_list):
    eval_model = model('gpt-4o-mini')
    generator = eval_model.gpt_generator

    for m in model_list:
        log = f'{log_path}/multilingual/{m}.json'
        with open(log, 'r', encoding='utf-8') as json_file:
            log_data = json.load(json_file)

        eval_logs_path = f'{log_path}/eval/multilingual/{m}.json'
        if os.path.exists(eval_logs_path):
            print(f'eval logs file already exists, will be skipped: {eval_logs_path}')
        else:
            eval_logs = []
            print(f'processing: {log}')
            with tqdm(total=len(log_data)) as pbar:
                for d in log_data:
                    try:
                        idx = d['idx']
                        answers = d['answers']
                        topic_difficulty = d['topic_difficulty']
                        question_type = d['question_type']
                        language = d['language']
                        is_arithmetic = d['is_arithmetic']

                        predict_answers_clp_en = d['clp_histories'][0]['answer']
                        correct_clp_en = 0
                        predict_answers_clp_zh = d['clp_histories'][1]['answer']
                        correct_clp_zh = 0
                        predict_answers_clp_fr = d['clp_histories'][2]['answer']
                        correct_clp_fr = 0

                        if question_type == 'multiple-choice':
                            if answers == 'c.':
                                answers = ['C']
                            elif answers == 'b.':
                                answers = ['B']
                            else:
                                answers = answers.split(', ')
                            predict_answers_clp_en = extract_options(predict_answers_clp_en)
                            predict_answers_clp_zh = extract_options(predict_answers_clp_zh)
                            predict_answers_clp_fr = extract_options(predict_answers_clp_fr)

                            is_equal = set(answers) == set(predict_answers_clp_en)
                            if is_equal:
                                correct_clp_en = 1
                            else:
                                correct_clp_en = 0
                            is_equal = set(answers) == set(predict_answers_clp_zh)
                            if is_equal:
                                correct_clp_zh = 1
                            else:
                                correct_clp_zh = 0
                            is_equal = set(answers) == set(predict_answers_clp_fr)
                            if is_equal:
                                correct_clp_fr = 1
                            else:
                                correct_clp_fr = 0

                        elif question_type == 'open question':
                            if predict_answers_clp_en == None:
                                correct_clp_en = 0
                            else:
                                prompt_clp_en = f'''Please evaluate whether the predicted answer is correct in comparison to the standard answer. If it is correct, output {{"correct": 1}}; if it is incorrect, output {{"correct": 0}}.\nPredicted answer: {predict_answers_clp_en}\nStandard answer: {answers}'''
                                response_clp_en, history_clp_en = generator(prompt_clp_en)
                                correct_clp_en = extract_response(response_clp_en)
                            
                            if predict_answers_clp_zh == None:
                                correct_clp_zh = 0
                            else:
                                prompt_clp_zh = f'''Please evaluate whether the predicted answer is correct in comparison to the standard answer. If it is correct, output {{"correct": 1}}; if it is incorrect, output {{"correct": 0}}.\nPredicted answer: {predict_answers_clp_zh}\nStandard answer: {answers}'''
                                response_clp_zh, history_clp_zh = generator(prompt_clp_zh)
                                correct_clp_zh = extract_response(response_clp_zh)

                            if predict_answers_clp_fr == None:
                                correct_clp_fr = 0
                            else:
                                prompt_clp_fr = f'''Please evaluate whether the predicted answer is correct in comparison to the standard answer. If it is correct, output {{"correct": 1}}; if it is incorrect, output {{"correct": 0}}.\nPredicted answer: {predict_answers_clp_fr}\nStandard answer: {answers}'''
                                response_clp_fr, history_clp_fr = generator(prompt_clp_fr)
                                correct_clp_fr = extract_response(response_clp_fr)
                        eval_log = {
                            "idx": idx,
                            "answers": answers,
                            "topic_difficulty": topic_difficulty,
                            "question_type": question_type,
                            "language": language,
                            "is_arithmetic": is_arithmetic,
                            "predict_answers_clp_en": predict_answers_clp_en,
                            "predict_answers_clp_zh": predict_answers_clp_zh,
                            "predict_answers_clp_fr": predict_answers_clp_fr,
                            "correct_clp_en": correct_clp_en,
                            "correct_clp_zh": correct_clp_zh,
                            "correct_clp_fr": correct_clp_fr
                        }
                        eval_logs.append(eval_log)
                        pbar.update(1)
                    except Exception:
                        eval_log = {
                            "idx": idx,
                            "answers": answers,
                            "topic_difficulty": topic_difficulty,
                            "question_type": question_type,
                            "language": language,
                            "is_arithmetic": is_arithmetic,
                            "predict_answers_clp_en": None,
                            "predict_answers_clp_zh": None,
                            "predict_answers_clp_fr": None,
                            "correct_clp_en": 0,
                            "correct_clp_zh": 0,
                            "correct_clp_fr": 0
                        }
                        # print(idx)
                    # break
                eval_logs_ = json.dumps(eval_logs, ensure_ascii=False)
                eval_logs_file = open(eval_logs_path, 'w', encoding='utf-8')
                eval_logs_file.write(eval_logs_)
                eval_logs_file.close()

def eval_translate(log_path, model_list):
    eval_model = model('gpt-4o-mini')
    generator = eval_model.gpt_generator

    for m in model_list:
        log = f'{log_path}/translate/{m}.json'
        with open(log, 'r', encoding='utf-8') as json_file:
            log_data = json.load(json_file)

        eval_logs_path = f'{log_path}/eval/translate/{m}.json'
        if os.path.exists(eval_logs_path):
            print(f'eval logs file already exists, will be skipped: {eval_logs_path}')
        else:
            eval_logs = []
            print(f'processing: {log}')
            with tqdm(total=len(log_data)) as pbar:
                for d in log_data:
                    idx = d['idx']
                    answers = d['answers']
                    topic_difficulty = d['topic_difficulty']
                    question_type = d['question_type']
                    language = d['language']
                    is_arithmetic = d['is_arithmetic']

                    predict_answers = d['history']['answer']
                    correct = 0

                    if question_type == 'multiple-choice':
                        if answers == 'c.':
                            answers = ['C']
                        elif answers == 'b.':
                            answers = ['B']
                        else:
                            answers = answers.split(', ')
                        predict_answers = extract_options(predict_answers)
                        is_equal = set(answers) == set(predict_answers)
                        if is_equal:
                            correct = 1
                        else:
                            correct = 0
                    elif question_type == 'open question':
                        prompt = f'''Please evaluate whether the predicted answer is correct in comparison to the standard answer. If it is correct, output {{"correct": 1}}; if it is incorrect, output {{"correct": 0}}.\nPredicted answer: {predict_answers}\nStandard answer: {answers}'''
                        response, history = generator(prompt)
                        correct = extract_response(response)
                    eval_log = {
                        "idx": idx,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "predict_answers": predict_answers,
                        "correct": correct
                    }
                    eval_logs.append(eval_log)
                    pbar.update(1)
                    # break
                eval_logs_ = json.dumps(eval_logs, ensure_ascii=False)
                eval_logs_file = open(eval_logs_path, 'w', encoding='utf-8')
                eval_logs_file.write(eval_logs_)
                eval_logs_file.close()

def main():
    args = parser.parse_args()
    log_path = f'{args.log_path}/eval/{args.eval}'
    check_dir(log_path)
    model_list = ['Meta-Llama-3.1-8B-Instruct', 'gpt-4o-mini', 'gemini-1.5-flash', 'qwen2.5-32b-instruct', 'deepseek-chat', 'DeepSeek-R1-Distill-Qwen-32B', 'Qwen3-14B', 'o4-mini']
    # model_list = ['o4-mini']

    prompting_method_list = ['direct', 'cot', 'ps']

    if args.eval == 'prompting':
        eval_prompting(args.log_path, model_list, prompting_method_list)
    elif args.eval == 'agent':
        eval_agent(args.log_path, model_list)
    elif args.eval == 'multilingual':
        eval_multilingual(args.log_path, model_list)
    elif args.eval == 'translate':
        eval_translate(args.log_path, model_list)

if __name__ == '__main__':
    main()