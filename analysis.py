from model import model
from prompt import *
import json
from tqdm import tqdm
import pandas as pd

class analysis():
    def __init__(self, log_path):
        self.log_path = log_path
        pass

    def _dataset_loader_(self, path=''):
        if path == '':
            parquet_file = './data/release_basic_txt-00000-of-00001.parquet'
        else:
            parquet_file = path
        df = pd.read_parquet(parquet_file)
        json_data = df.to_json(orient='records')
        return json.loads(json_data)
    
    def _dataset_context_id_(self, json_data):
        context_idx = {}
        for i in range(len(json_data)):
            data = json_data[i]
            if int(data['sub_question_id']) == 1:
                context_idx[f'{data["language"]}_{data["main_question_id"]}'.format()] = i
        return context_idx
    
    def prompting_analysis(self, eval_model_name, infra_api='deepinfra'):
        # read dataset
        dataset = self._dataset_loader_()
        context_idx = self._dataset_context_id_(dataset)

        # model init
        print(f'eval model: {eval_model_name}, infra api: {infra_api}')
        eval_model = model(eval_model_name)
        if infra_api == 'deepinfra':
            generator = eval_model.deepinfra_generator
        elif infra_api == 'openai':
            generator = eval_model.gpt_generator
        elif infra_api == 'deepseek':
            generator = eval_model.deepseek_generator
        elif infra_api == 'aliyun':
            generator = eval_model.aliyun_generator

        # logs init
        logs_direct = []
        logs_cot = []
        logs_ps = []

        n = len(dataset)
        with tqdm(total=n) as pbar:
            for i in range(n):
                try:
                    data = dataset[i]

                    idx = data['idx']
                    question_id = data['question_id']
                    if data['context'] == 'nan':
                        context = dataset[context_idx[f'{data["language"]}_{data["main_question_id"]}']]['context']
                    else:
                        context = data['context']
                    question = data['question']
                    options = data['options']
                    image_type = data['image_type']
                    answers = data['answers']
                    topic_difficulty = data['topic_difficulty']
                    question_type = data['question_type']
                    subfield = data['subfield']
                    language = data['language']
                    if language == 'english':
                        language = 'English'
                    elif language == 'chinese':
                        language = 'Chinese'
                    elif language == 'french':
                        language = 'French'
                    is_arithmetic = data['is_arithmetic']

                    if question_type == 'open question':
                        input = f'context: {context}\n\nquestion: {question}'
                    elif question_type == 'multiple-choice':
                        input = f'context: {context}\n\nquestion: {question}\n\noptions: {options}'

                    ##################################################
                    # Direct
                    input_direct = get_base_prompt(input, language, question_type)
                    output_direct, history_direct = generator(input_direct)
                    answer_direct = history_direct[-1]['content']
                    log_direct_history = {
                        "method": "direct",
                        "history": history_direct,
                        "answer": answer_direct,
                    }
                    log_direct = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": log_direct_history
                    }
                    ##################################################
                    # Zero-shot CoT
                    input_cot = get_CoT_prompt(input, language, question_type)
                    output_cot, history_cot = generator(input_cot)
                    answer_cot = history_cot[-1]['content']
                    log_cot_history = {
                        "method": "cot",
                        "history": history_cot,
                        "answer": answer_cot,
                    }
                    log_cot = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": log_cot_history
                    }
                    ##################################################
                    # Plan-and-Solve
                    input_ps = get_ps_prompt(input, language, question_type)
                    output_ps, history_ps = generator(input_ps)
                    answer_ps = history_ps[-1]['content']
                    log_ps_history = {
                        "method": "ps",
                        "history": history_ps,
                        "answer": answer_ps,
                    }
                    log_ps = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": log_ps_history
                    }
                    ##################################################
                except Exception as e:
                    # print(e)
                    log_direct, log_cot, log_ps = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": []
                    }
                logs_direct.append(log_direct)
                logs_cot.append(log_cot)
                logs_ps.append(log_ps)

                pbar.update(1)
                # break
                if pbar.n % 10 == 0:
                    logs_direct_ = json.dumps(logs_direct, ensure_ascii=False)
                    logs_cot_ = json.dumps(logs_cot, ensure_ascii=False)
                    logs_ps_ = json.dumps(logs_ps, ensure_ascii=False)

                    log_file_direct = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-direct.json', 'w', encoding='utf-8')
                    log_file_cot = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-cot.json', 'w', encoding='utf-8')
                    log_file_ps = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-ps.json', 'w', encoding='utf-8')

                    log_file_direct.write(logs_direct_)
                    log_file_cot.write(logs_cot_)
                    log_file_ps.write(logs_ps_)

                    log_file_direct.close()
                    log_file_cot.close()
                    log_file_ps.close()
        logs_direct_ = json.dumps(logs_direct, ensure_ascii=False)
        logs_cot_ = json.dumps(logs_cot, ensure_ascii=False)
        logs_ps_ = json.dumps(logs_ps, ensure_ascii=False)

        log_file_direct = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-direct.json', 'w', encoding='utf-8')
        log_file_cot = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-cot.json', 'w', encoding='utf-8')
        log_file_ps = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-ps.json', 'w', encoding='utf-8')

        log_file_direct.write(logs_direct_)
        log_file_cot.write(logs_cot_)
        log_file_ps.write(logs_ps_)

        log_file_direct.close()
        log_file_cot.close()
        log_file_ps.close()
    
    def translate_analysis(self, eval_model_name, infra_api='deepinfra'):
        # read dataset
        with open('./data/basic_fr.json', 'r', encoding='utf-8') as json_file_fr:
            data_fr = json.load(json_file_fr)
        with open('./data/basic_zh.json', 'r', encoding='utf-8') as json_file_zh:
            data_zh = json.load(json_file_zh)
        dataset = data_fr + data_zh

        # model init
        print(f'eval model: {eval_model_name}, infra api: {infra_api}')
        eval_model = model(eval_model_name)
        if infra_api == 'deepinfra':
            generator = eval_model.deepinfra_generator
        elif infra_api == 'openai':
            generator = eval_model.gpt_generator
        elif infra_api == 'deepseek':
            generator = eval_model.deepseek_generator
        elif infra_api == 'aliyun':
            generator = eval_model.aliyun_generator

        # logs init
        logs_direct = []

        n = len(dataset)
        with tqdm(total=n) as pbar:
            for i in range(n):
                try:
                    data = dataset[i]

                    idx = data['idx']
                    question_id = data['question_id']
                    context = data['context']
                    question = data['question']
                    options = data['options']
                    image_type = data['image_type']
                    answers = data['answers']
                    topic_difficulty = data['topic_difficulty']
                    question_type = data['question_type']
                    subfield = data['subfield']
                    language = data['language']
                    is_arithmetic = data['is_arithmetic']
                    translate_question = data['translate_question']
                    translate_context = data['translate_context']

                    if question_type == 'open question':
                        input = f'context: {context}\n\nquestion: {question}'
                    elif question_type == 'multiple-choice':
                        input = f'context: {context}\n\nquestion: {question}\n\noptions: {options}'

                    ##################################################
                    input_direct = get_base_prompt(input, language, question_type)
                    output_direct, history_direct = generator(input_direct)
                    answer_direct = history_direct[-1]['content']
                    log_direct_history = {
                        "method": "direct",
                        "history": history_direct,
                        "answer": answer_direct,
                    }
                    log_direct = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": log_direct_history
                    }
                    ##################################################
                except Exception as e:
                    # print(e)
                    log_direct = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": {}
                    }
                logs_direct.append(log_direct)

                pbar.update(1)
                # break
                if pbar.n % 10 == 0:
                    logs_direct_ = json.dumps(logs_direct, ensure_ascii=False)
                    log_file_direct = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}.json', 'w', encoding='utf-8')
                    log_file_direct.write(logs_direct_)
                    log_file_direct.close()
        logs_direct_ = json.dumps(logs_direct, ensure_ascii=False)
        log_file_direct = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}.json', 'w', encoding='utf-8')
        log_file_direct.write(logs_direct_)
        log_file_direct.close()

    def agent_analysis(self, eval_model_name, infra_api='deepinfra'):
        # read dataset
        dataset = self._dataset_loader_()
        context_idx = self._dataset_context_id_(dataset)

        # model init
        print(f'eval model: {eval_model_name}, infra api: {infra_api}')
        eval_model = model(eval_model_name)
        if infra_api == 'deepinfra':
            generator = eval_model.deepinfra_generator
        elif infra_api == 'openai':
            generator = eval_model.gpt_generator
        elif infra_api == 'deepseek':
            generator = eval_model.deepseek_generator
        elif infra_api == 'aliyun':
            generator = eval_model.aliyun_generator

        # logs init
        logs_agent = []

        n = len(dataset)
        with tqdm(total=n) as pbar:
            for i in range(n):
                try:
                    data = dataset[i]

                    idx = data['idx']
                    question_id = data['question_id']
                    if data['context'] == 'nan':
                        context = dataset[context_idx[f'{data["language"]}_{data["main_question_id"]}']]['context']
                    else:
                        context = data['context']
                    question = data['question']
                    options = data['options']
                    image_type = data['image_type']
                    answers = data['answers']
                    topic_difficulty = data['topic_difficulty']
                    question_type = data['question_type']
                    subfield = data['subfield']
                    language = data['language']
                    if language == 'english':
                        language = 'English'
                    elif language == 'chinese':
                        language = 'Chinese'
                    elif language == 'french':
                        language = 'French'
                    is_arithmetic = data['is_arithmetic']

                    if question_type == 'open question':
                        input = f'context: {context}\n\nquestion: {question}'
                    elif question_type == 'multiple-choice':
                        input = f'context: {context}\n\nquestion: {question}\n\noptions: {options}'

                    ##################################################
                    # Self-Refine
                    input_answer = get_self_refine_answer_prompt(input, language, question_type)
                    output_answer, history_answer = generator(input_answer)

                    input_feedback_ = f'{input}\n\nanswer: {output_answer}'
                    input_feedback = get_self_refine_feedback_prompt(input_feedback_)
                    output_feedback, history_feedback = generator(input_feedback)

                    input_refine_ = f'current answer: {output_answer}\n\nsuggestions: {output_feedback}'
                    input_refine = get_self_refine_refine_prompt(input_refine_, language, question_type)
                    output_refine, history_refine = generator(input_refine)

                    log_self_refine_output = {
                        "output_answer": output_answer,
                        "output_feedback": output_feedback,
                        "output_refine": output_refine
                    }
                    ##################################################
                    # S3 Agent
                    input_s3_agent_1 = get_s3agent_1_prompt(input, language, question_type)
                    output_1, history_1 = generator(input_s3_agent_1)

                    input_s3_agent_2 = get_s3agent_2_prompt(input, language, question_type)
                    output_2, history_2 = generator(input_s3_agent_2)

                    input_s3_agent_3 = get_s3agent_3_prompt(input, language, question_type)
                    output_3, history_3 = generator(input_s3_agent_3)

                    input_final_ = f'Perspectives:\nSuperficial Expression: {output_1}\nSemantic Information: {output_2}\nSentiment Expression: {output_3}'
                    input_s3_agent_final = get_s3agent_final_prompt(input_final_, language, question_type)
                    output_final, history_final = generator(input_s3_agent_final)

                    log_s3_agent_output = {
                        "s3_agent_1": output_1,
                        "s3_agent_2": output_2,
                        "s3_agent_3": output_3,
                        "s3_agent_final": output_final
                    }
                    ##################################################
                    log_agent = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "self_refine_output": log_self_refine_output,
                        "s3_agent_output": log_s3_agent_output
                    }
                except Exception as e:
                    # print(e)
                    log_agent = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "self_refine_output": {},
                        "s3_agent_output": {}
                    }
                logs_agent.append(log_agent)

                pbar.update(1)
                # break
                if pbar.n % 10 == 0:
                    logs_agent_ = json.dumps(logs_agent, ensure_ascii=False)
                    log_file_agent = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}.json', 'w', encoding='utf-8')
                    log_file_agent.write(logs_agent_)
                    log_file_agent.close()
        logs_agent_ = json.dumps(logs_agent, ensure_ascii=False)
        log_file_agent = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}.json', 'w', encoding='utf-8')
        log_file_agent.write(logs_agent_)
        log_file_agent.close()

    def multilingual_analysis(self, eval_model_name, infra_api='deepinfra', only_zh_fr=False):
        # read dataset
        if only_zh_fr:
            with open('./data/basic_fr.json', 'r', encoding='utf-8') as json_file_fr:
                data_fr = json.load(json_file_fr)
            with open('./data/basic_zh.json', 'r', encoding='utf-8') as json_file_zh:
                data_zh = json.load(json_file_zh)
            dataset = data_fr + data_zh
        else:
            dataset = self._dataset_loader_()
            context_idx = self._dataset_context_id_(dataset)

        # model init
        print(f'eval model: {eval_model_name}, infra api: {infra_api}')
        eval_model = model(eval_model_name)
        if infra_api == 'deepinfra':
            generator = eval_model.deepinfra_generator
        elif infra_api == 'openai':
            generator = eval_model.gpt_generator
        elif infra_api == 'deepseek':
            generator = eval_model.deepseek_generator
        elif infra_api == 'aliyun':
            generator = eval_model.aliyun_generator

        # logs init
        logs = []

        n = len(dataset)
        with tqdm(total=n) as pbar:
            for i in range(n):
                try:
                    data = dataset[i]

                    idx = data['idx']
                    question_id = data['question_id']
                    if data['context'] == 'nan':
                        context = dataset[context_idx[f'{data["language"]}_{data["main_question_id"]}']]['context']
                    else:
                        context = data['context']
                    question = data['question']
                    options = data['options']
                    image_type = data['image_type']
                    answers = data['answers']
                    topic_difficulty = data['topic_difficulty']
                    question_type = data['question_type']
                    subfield = data['subfield']
                    language = data['language']
                    if language == 'english':
                        language = 'English'
                    elif language == 'chinese':
                        language = 'Chinese'
                    elif language == 'french':
                        language = 'French'
                    is_arithmetic = data['is_arithmetic']

                    if question_type == 'open question':
                        input = f'context: {context}\n\nquestion: {question}'
                    elif question_type == 'multiple-choice':
                        input = f'context: {context}\n\nquestion: {question}\n\noptions: {options}'

                    clp_languages = ['English', 'Chinese', 'French']
                    clp_histories = []
                    ##################################################
                    for clp_language in clp_languages:
                        # CLP -> clp_language
                        input_stage1 = get_clp_stage1_prompt(input, clp_language, language)
                        # CLP Stage 1
                        output, history = generator(input_stage1)
                        # CLP Stage 2
                        input_stage2 = get_clp_stage2_prompt(clp_language, language, question_type)
                        output, history = generator(input_stage2, history)
                        answer = history[-1]['content']
                        clp_history = {
                            "clp_language": clp_language,
                            "source_language": language,
                            "history": history,
                            "answer": answer,
                            # "is_correct": -1 # default number
                        }
                        clp_histories.append(clp_history)
                        if only_zh_fr:
                            clp_history_zh = {
                            "clp_language": "Chinese",
                            "source_language": language,
                            "history": [],
                            "answer": None,
                            # "is_correct": -1 # default number
                            }
                            clp_histories.append(clp_history_zh)
                            clp_history_fr = {
                            "clp_language": "French",
                            "source_language": language,
                            "history": [],
                            "answer": None,
                            # "is_correct": -1 # default number
                            }
                            clp_histories.append(clp_history_fr)
                            break
                    ##################################################
                    log = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "clp_histories": clp_histories
                    }
                except Exception as e:
                    # print(e)
                    log = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "clp_histories": []
                    }
                logs.append(log)
                pbar.update(1)
                # break
                if pbar.n % 10 == 0:
                    logs_ = json.dumps(logs, ensure_ascii=False)
                    log_file = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}.json', 'w', encoding='utf-8')
                    log_file.write(logs_)
                    log_file.close()
        logs = json.dumps(logs, ensure_ascii=False)
        log_file = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}.json', 'w', encoding='utf-8')
        log_file.write(logs)
        log_file.close()

    def prompting_analysis_non_thinking(self, eval_model_name, infra_api='deepinfra'):
        # read dataset
        dataset = self._dataset_loader_()
        context_idx = self._dataset_context_id_(dataset)

        # model init
        print(f'eval model: {eval_model_name}, infra api: {infra_api}')
        eval_model = model(eval_model_name)
        if infra_api == 'deepinfra':
            print('input_extra_body ERROR')
        elif infra_api == 'openai':
            print('input_extra_body ERROR')
        elif infra_api == 'deepseek':
            print('input_extra_body ERROR')
        elif infra_api == 'aliyun':
            generator = eval_model.aliyun_generator

        # logs init
        logs_direct = []
        logs_cot = []
        logs_ps = []

        n = len(dataset)
        with tqdm(total=n) as pbar:
            for i in range(n):
                try:
                    data = dataset[i]

                    idx = data['idx']
                    question_id = data['question_id']
                    if data['context'] == 'nan':
                        context = dataset[context_idx[f'{data["language"]}_{data["main_question_id"]}']]['context']
                    else:
                        context = data['context']
                    question = data['question']
                    options = data['options']
                    image_type = data['image_type']
                    answers = data['answers']
                    topic_difficulty = data['topic_difficulty']
                    question_type = data['question_type']
                    subfield = data['subfield']
                    language = data['language']
                    if language == 'english':
                        language = 'English'
                    elif language == 'chinese':
                        language = 'Chinese'
                    elif language == 'french':
                        language = 'French'
                    is_arithmetic = data['is_arithmetic']

                    if question_type == 'open question':
                        input = f'context: {context}\n\nquestion: {question}'
                    elif question_type == 'multiple-choice':
                        input = f'context: {context}\n\nquestion: {question}\n\noptions: {options}'

                    ##################################################
                    # Direct
                    input_direct = get_base_prompt(input, language, question_type)
                    output_direct, history_direct = generator(input_direct, input_extra_body={"enable_thinking": False})
                    answer_direct = history_direct[-1]['content']
                    log_direct_history = {
                        "method": "direct",
                        "history": history_direct,
                        "answer": answer_direct,
                    }
                    log_direct = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": log_direct_history
                    }
                    ##################################################
                    # Zero-shot CoT
                    input_cot = get_CoT_prompt(input, language, question_type)
                    output_cot, history_cot = generator(input_cot, input_extra_body={"enable_thinking": False})
                    answer_cot = history_cot[-1]['content']
                    log_cot_history = {
                        "method": "cot",
                        "history": history_cot,
                        "answer": answer_cot,
                    }
                    log_cot = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": log_cot_history
                    }
                    ##################################################
                    # Plan-and-Solve
                    input_ps = get_ps_prompt(input, language, question_type)
                    output_ps, history_ps = generator(input_ps, input_extra_body={"enable_thinking": False})
                    answer_ps = history_ps[-1]['content']
                    log_ps_history = {
                        "method": "ps",
                        "history": history_ps,
                        "answer": answer_ps,
                    }
                    log_ps = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": log_ps_history
                    }
                    ##################################################
                except Exception as e:
                    # print(e)
                    log_direct, log_cot, log_ps = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": []
                    }
                logs_direct.append(log_direct)
                logs_cot.append(log_cot)
                logs_ps.append(log_ps)

                pbar.update(1)
                # break
                if pbar.n % 10 == 0:
                    logs_direct_ = json.dumps(logs_direct, ensure_ascii=False)
                    logs_cot_ = json.dumps(logs_cot, ensure_ascii=False)
                    logs_ps_ = json.dumps(logs_ps, ensure_ascii=False)

                    log_file_direct = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-direct-non-thinking.json', 'w', encoding='utf-8')
                    log_file_cot = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-cot-non-thinking.json', 'w', encoding='utf-8')
                    log_file_ps = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-ps-non-thinking.json', 'w', encoding='utf-8')

                    log_file_direct.write(logs_direct_)
                    log_file_cot.write(logs_cot_)
                    log_file_ps.write(logs_ps_)

                    log_file_direct.close()
                    log_file_cot.close()
                    log_file_ps.close()
        logs_direct_ = json.dumps(logs_direct, ensure_ascii=False)
        logs_cot_ = json.dumps(logs_cot, ensure_ascii=False)
        logs_ps_ = json.dumps(logs_ps, ensure_ascii=False)

        log_file_direct = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-direct-non-thinking.json', 'w', encoding='utf-8')
        log_file_cot = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-cot-non-thinking.json', 'w', encoding='utf-8')
        log_file_ps = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-ps-non-thinking.json', 'w', encoding='utf-8')

        log_file_direct.write(logs_direct_)
        log_file_cot.write(logs_cot_)
        log_file_ps.write(logs_ps_)

        log_file_direct.close()
        log_file_cot.close()
        log_file_ps.close()

    def prompting_analysis_direct(self, eval_model_name, infra_api='deepinfra'):
        # read dataset
        dataset = self._dataset_loader_()
        context_idx = self._dataset_context_id_(dataset)

        # model init
        print(f'eval model: {eval_model_name}, infra api: {infra_api}')
        eval_model = model(eval_model_name)
        if infra_api == 'deepinfra':
            generator = eval_model.deepinfra_generator
        elif infra_api == 'openai':
            generator = eval_model.gpt_generator
        elif infra_api == 'deepseek':
            generator = eval_model.deepseek_generator
        elif infra_api == 'aliyun':
            generator = eval_model.aliyun_generator

        # logs init
        logs_direct = []
        logs_cot = []
        logs_ps = []

        n = len(dataset)
        with tqdm(total=n) as pbar:
            for i in range(n):
                try:
                    data = dataset[i]

                    idx = data['idx']
                    question_id = data['question_id']
                    if data['context'] == 'nan':
                        context = dataset[context_idx[f'{data["language"]}_{data["main_question_id"]}']]['context']
                    else:
                        context = data['context']
                    question = data['question']
                    options = data['options']
                    image_type = data['image_type']
                    answers = data['answers']
                    topic_difficulty = data['topic_difficulty']
                    question_type = data['question_type']
                    subfield = data['subfield']
                    language = data['language']
                    if language == 'english':
                        language = 'English'
                    elif language == 'chinese':
                        language = 'Chinese'
                    elif language == 'french':
                        language = 'French'
                    is_arithmetic = data['is_arithmetic']

                    if question_type == 'open question':
                        input = f'context: {context}\n\nquestion: {question}'
                    elif question_type == 'multiple-choice':
                        input = f'context: {context}\n\nquestion: {question}\n\noptions: {options}'

                    ##################################################
                    # Direct
                    # input_direct = get_base_prompt(input, language, question_type)
                    # output_direct, history_direct = generator(input_direct)
                    # answer_direct = history_direct[-1]['content']
                    # log_direct_history = {
                    #     "method": "direct",
                    #     "history": history_direct,
                    #     "answer": answer_direct,
                    # }
                    # log_direct = {
                    #     "idx": idx,
                    #     "question_id": question_id,
                    #     "context": context,
                    #     "question": question,
                    #     "options": options,
                    #     "image_type": image_type,
                    #     "answers": answers,
                    #     "topic_difficulty": topic_difficulty,
                    #     "question_type": question_type,
                    #     "subfield": subfield,
                    #     "language": language,
                    #     "is_arithmetic": is_arithmetic,
                    #     "history": log_direct_history
                    # }
                    ##################################################
                    # Zero-shot CoT
                    input_cot = get_CoT_prompt(input, language, question_type)
                    output_cot, history_cot = generator(input_cot)
                    answer_cot = history_cot[-1]['content']
                    log_cot_history = {
                        "method": "cot",
                        "history": history_cot,
                        "answer": answer_cot,
                    }
                    log_cot = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": log_cot_history
                    }
                    ##################################################
                    # Plan-and-Solve
                    input_ps = get_ps_prompt(input, language, question_type)
                    output_ps, history_ps = generator(input_ps)
                    answer_ps = history_ps[-1]['content']
                    log_ps_history = {
                        "method": "ps",
                        "history": history_ps,
                        "answer": answer_ps,
                    }
                    log_ps = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": log_ps_history
                    }
                    ##################################################
                except Exception as e:
                    # print(e)
                    log_direct, log_cot, log_ps = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": []
                    }
                # logs_direct.append(log_direct)
                logs_cot.append(log_cot)
                logs_ps.append(log_ps)

                pbar.update(1)
                # break
                if pbar.n % 10 == 0:
                    # logs_direct_ = json.dumps(logs_direct, ensure_ascii=False)
                    logs_cot_ = json.dumps(logs_cot, ensure_ascii=False)
                    logs_ps_ = json.dumps(logs_ps, ensure_ascii=False)

                    # log_file_direct = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-direct.json', 'w', encoding='utf-8')
                    log_file_cot = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-cot.json', 'w', encoding='utf-8')
                    log_file_ps = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-ps.json', 'w', encoding='utf-8')

                    # log_file_direct.write(logs_direct_)
                    log_file_cot.write(logs_cot_)
                    log_file_ps.write(logs_ps_)

                    # log_file_direct.close()
                    log_file_cot.close()
                    log_file_ps.close()
        # logs_direct_ = json.dumps(logs_direct, ensure_ascii=False)
        logs_cot_ = json.dumps(logs_cot, ensure_ascii=False)
        logs_ps_ = json.dumps(logs_ps, ensure_ascii=False)

        # log_file_direct = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-direct.json', 'w', encoding='utf-8')
        log_file_cot = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-cot.json', 'w', encoding='utf-8')
        log_file_ps = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-ps.json', 'w', encoding='utf-8')

        # log_file_direct.write(logs_direct_)
        log_file_cot.write(logs_cot_)
        log_file_ps.write(logs_ps_)

        # log_file_direct.close()
        log_file_cot.close()
        log_file_ps.close()

    def prompting_analysis_direct_non_thinking(self, eval_model_name, infra_api='deepinfra'):
        # read dataset
        dataset = self._dataset_loader_()
        context_idx = self._dataset_context_id_(dataset)

        # model init
        print(f'eval model: {eval_model_name}, infra api: {infra_api}')
        eval_model = model(eval_model_name)
        if infra_api == 'deepinfra':
            generator = eval_model.deepinfra_generator
        elif infra_api == 'openai':
            generator = eval_model.gpt_generator
        elif infra_api == 'deepseek':
            generator = eval_model.deepseek_generator
        elif infra_api == 'aliyun':
            generator = eval_model.aliyun_generator

        # logs init
        logs_direct = []
        logs_cot = []
        logs_ps = []

        n = len(dataset)
        with tqdm(total=n) as pbar:
            for i in range(n):
                try:
                    data = dataset[i]

                    idx = data['idx']
                    question_id = data['question_id']
                    if data['context'] == 'nan':
                        context = dataset[context_idx[f'{data["language"]}_{data["main_question_id"]}']]['context']
                    else:
                        context = data['context']
                    question = data['question']
                    options = data['options']
                    image_type = data['image_type']
                    answers = data['answers']
                    topic_difficulty = data['topic_difficulty']
                    question_type = data['question_type']
                    subfield = data['subfield']
                    language = data['language']
                    if language == 'english':
                        language = 'English'
                    elif language == 'chinese':
                        language = 'Chinese'
                    elif language == 'french':
                        language = 'French'
                    is_arithmetic = data['is_arithmetic']

                    if question_type == 'open question':
                        input = f'context: {context}\n\nquestion: {question}'
                    elif question_type == 'multiple-choice':
                        input = f'context: {context}\n\nquestion: {question}\n\noptions: {options}'

                    ##################################################
                    # Direct
                    input_direct = get_base_prompt(input, language, question_type)
                    output_direct, history_direct = generator(input_direct, input_extra_body={"enable_thinking": False})
                    answer_direct = history_direct[-1]['content']
                    log_direct_history = {
                        "method": "direct",
                        "history": history_direct,
                        "answer": answer_direct,
                    }
                    log_direct = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": log_direct_history
                    }
                    # ##################################################
                    # # Zero-shot CoT
                    # input_cot = get_CoT_prompt(input, language, question_type)
                    # output_cot, history_cot = generator(input_cot)
                    # answer_cot = history_cot[-1]['content']
                    # log_cot_history = {
                    #     "method": "cot",
                    #     "history": history_cot,
                    #     "answer": answer_cot,
                    # }
                    # log_cot = {
                    #     "idx": idx,
                    #     "question_id": question_id,
                    #     "context": context,
                    #     "question": question,
                    #     "options": options,
                    #     "image_type": image_type,
                    #     "answers": answers,
                    #     "topic_difficulty": topic_difficulty,
                    #     "question_type": question_type,
                    #     "subfield": subfield,
                    #     "language": language,
                    #     "is_arithmetic": is_arithmetic,
                    #     "history": log_cot_history
                    # }
                    # ##################################################
                    # # Plan-and-Solve
                    # input_ps = get_ps_prompt(input, language, question_type)
                    # output_ps, history_ps = generator(input_ps)
                    # answer_ps = history_ps[-1]['content']
                    # log_ps_history = {
                    #     "method": "ps",
                    #     "history": history_ps,
                    #     "answer": answer_ps,
                    # }
                    # log_ps = {
                    #     "idx": idx,
                    #     "question_id": question_id,
                    #     "context": context,
                    #     "question": question,
                    #     "options": options,
                    #     "image_type": image_type,
                    #     "answers": answers,
                    #     "topic_difficulty": topic_difficulty,
                    #     "question_type": question_type,
                    #     "subfield": subfield,
                    #     "language": language,
                    #     "is_arithmetic": is_arithmetic,
                    #     "history": log_ps_history
                    # }
                    # ##################################################
                except Exception as e:
                    # print(e)
                    log_direct, log_cot, log_ps = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "history": []
                    }
                logs_direct.append(log_direct)
                # logs_cot.append(log_cot)
                # logs_ps.append(log_ps)

                pbar.update(1)
                # break
                if pbar.n % 10 == 0:
                    logs_direct_ = json.dumps(logs_direct, ensure_ascii=False)
                    # logs_cot_ = json.dumps(logs_cot, ensure_ascii=False)
                    # logs_ps_ = json.dumps(logs_ps, ensure_ascii=False)

                    log_file_direct = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-direct-non-thinking.json', 'w', encoding='utf-8')
                    # log_file_cot = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-cot.json', 'w', encoding='utf-8')
                    # log_file_ps = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-ps.json', 'w', encoding='utf-8')

                    log_file_direct.write(logs_direct_)
                    # log_file_cot.write(logs_cot_)
                    # log_file_ps.write(logs_ps_)

                    log_file_direct.close()
                    # log_file_cot.close()
                    # log_file_ps.close()
        logs_direct_ = json.dumps(logs_direct, ensure_ascii=False)
        # logs_cot_ = json.dumps(logs_cot, ensure_ascii=False)
        # logs_ps_ = json.dumps(logs_ps, ensure_ascii=False)

        log_file_direct = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-direct-non-thinking.json', 'w', encoding='utf-8')
        # log_file_cot = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-cot.json', 'w', encoding='utf-8')
        # log_file_ps = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-ps.json', 'w', encoding='utf-8')

        log_file_direct.write(logs_direct_)
        # log_file_cot.write(logs_cot_)
        # log_file_ps.write(logs_ps_)

        log_file_direct.close()
        # log_file_cot.close()
        # log_file_ps.close()

    def agent_analysis_non_thinking(self, eval_model_name, infra_api='deepinfra'):
        # read dataset
        dataset = self._dataset_loader_()
        context_idx = self._dataset_context_id_(dataset)

        # model init
        print(f'eval model: {eval_model_name}, infra api: {infra_api}')
        eval_model = model(eval_model_name)
        if infra_api == 'deepinfra':
            print('input_extra_body ERROR')
        elif infra_api == 'openai':
            print('input_extra_body ERROR')
        elif infra_api == 'deepseek':
            print('input_extra_body ERROR')
        elif infra_api == 'aliyun':
            generator = eval_model.aliyun_generator

        # logs init
        logs_agent = []

        n = len(dataset)
        with tqdm(total=n) as pbar:
            for i in range(n):
                try:
                    data = dataset[i]

                    idx = data['idx']
                    question_id = data['question_id']
                    if data['context'] == 'nan':
                        context = dataset[context_idx[f'{data["language"]}_{data["main_question_id"]}']]['context']
                    else:
                        context = data['context']
                    question = data['question']
                    options = data['options']
                    image_type = data['image_type']
                    answers = data['answers']
                    topic_difficulty = data['topic_difficulty']
                    question_type = data['question_type']
                    subfield = data['subfield']
                    language = data['language']
                    if language == 'english':
                        language = 'English'
                    elif language == 'chinese':
                        language = 'Chinese'
                    elif language == 'french':
                        language = 'French'
                    is_arithmetic = data['is_arithmetic']

                    if question_type == 'open question':
                        input = f'context: {context}\n\nquestion: {question}'
                    elif question_type == 'multiple-choice':
                        input = f'context: {context}\n\nquestion: {question}\n\noptions: {options}'

                    ##################################################
                    # Self-Refine
                    input_answer = get_self_refine_answer_prompt(input, language, question_type)
                    output_answer, history_answer = generator(input_answer, input_extra_body={"enable_thinking": False})

                    input_feedback_ = f'{input}\n\nanswer: {output_answer}'
                    input_feedback = get_self_refine_feedback_prompt(input_feedback_)
                    output_feedback, history_feedback = generator(input_feedback, input_extra_body={"enable_thinking": False})

                    input_refine_ = f'current answer: {output_answer}\n\nsuggestions: {output_feedback}'
                    input_refine = get_self_refine_refine_prompt(input_refine_, language, question_type)
                    output_refine, history_refine = generator(input_refine, input_extra_body={"enable_thinking": False})

                    log_self_refine_output = {
                        "output_answer": output_answer,
                        "output_feedback": output_feedback,
                        "output_refine": output_refine
                    }
                    ##################################################
                    # S3 Agent
                    input_s3_agent_1 = get_s3agent_1_prompt(input, language, question_type)
                    output_1, history_1 = generator(input_s3_agent_1, input_extra_body={"enable_thinking": False})

                    input_s3_agent_2 = get_s3agent_2_prompt(input, language, question_type)
                    output_2, history_2 = generator(input_s3_agent_2, input_extra_body={"enable_thinking": False})

                    input_s3_agent_3 = get_s3agent_3_prompt(input, language, question_type)
                    output_3, history_3 = generator(input_s3_agent_3, input_extra_body={"enable_thinking": False})

                    input_final_ = f'Perspectives:\nSuperficial Expression: {output_1}\nSemantic Information: {output_2}\nSentiment Expression: {output_3}'
                    input_s3_agent_final = get_s3agent_final_prompt(input_final_, language, question_type)
                    output_final, history_final = generator(input_s3_agent_final, input_extra_body={"enable_thinking": False})

                    log_s3_agent_output = {
                        "s3_agent_1": output_1,
                        "s3_agent_2": output_2,
                        "s3_agent_3": output_3,
                        "s3_agent_final": output_final
                    }
                    ##################################################
                    log_agent = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "self_refine_output": log_self_refine_output,
                        "s3_agent_output": log_s3_agent_output
                    }
                except Exception as e:
                    # print(e)
                    log_agent = {
                        "idx": idx,
                        "question_id": question_id,
                        "context": context,
                        "question": question,
                        "options": options,
                        "image_type": image_type,
                        "answers": answers,
                        "topic_difficulty": topic_difficulty,
                        "question_type": question_type,
                        "subfield": subfield,
                        "language": language,
                        "is_arithmetic": is_arithmetic,
                        "self_refine_output": {},
                        "s3_agent_output": {}
                    }
                logs_agent.append(log_agent)

                pbar.update(1)
                # break
                if pbar.n % 10 == 0:
                    logs_agent_ = json.dumps(logs_agent, ensure_ascii=False)
                    log_file_agent = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-non-thinking.json', 'w', encoding='utf-8')
                    log_file_agent.write(logs_agent_)
                    log_file_agent.close()
        logs_agent_ = json.dumps(logs_agent, ensure_ascii=False)
        log_file_agent = open(f'{self.log_path}/{eval_model_name.split("/")[-1]}-non-thinking.json', 'w', encoding='utf-8')
        log_file_agent.write(logs_agent_)
        log_file_agent.close()