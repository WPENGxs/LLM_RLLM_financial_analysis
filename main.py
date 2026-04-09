from analysis import analysis
import argparse
import os

parser = argparse.ArgumentParser()

# analysis = []
parser.add_argument('--analysis', type=str, default='')
parser.add_argument('--model', type=str, default='gpt-4o-mini')
parser.add_argument('--non_thinking', type='store_true', default=False)
parser.add_argument('--log_path', type=str, default='./log')

def check_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"An error occurred while creating path '{path}': {e.strerror}")

def main():
    args = parser.parse_args()
    model = args.model
    log_path = f'{args.log_path}/{args.analysis}'
    check_dir(log_path)

    if model == 'gpt-4o-mini':
        infra_model = 'gpt-4o-mini'
        infra_api = 'openai'
    elif model == 'llama-3.1-8b':
        infra_model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        infra_api = 'deepinfra'
    elif model == 'gemini-1.5-flash':
        infra_model = 'google/gemini-1.5-flash'
        infra_api = 'deepinfra'
    elif model == 'qwen-2.5-32b':
        infra_model = 'qwen2.5-32b-instruct'
        infra_api = 'aliyun'
    elif model == 'deepseek-v3':
        infra_model = 'deepseek-chat'
        infra_api = 'deepseek'
    elif model == 'deepseek-r1-dis':
        infra_model = 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
        infra_api = 'deepinfra'
    elif model == 'qwen3-14b':
        infra_model = 'qwen3-14b'
        infra_api = 'aliyun'
    elif model == 'o4-mini':
        infra_model = 'o4-mini'
        infra_api = 'openai'
    else:
        raise ValueError('Unknown Model')

    analysis_eval = analysis(log_path)

    if args.analysis == 'multilingual':
        print(f'start analysis: multilingual')
        analysis_eval.multilingual_analysis(infra_model, infra_api=infra_api)
        # analysis_eval.multilingual_analysis(infra_model, infra_api=infra_api, only_zh_fr=True)
    elif args.analysis == 'prompting':
        print(f'start analysis: prompting')
        if args.non_thinking:
            if model == 'qwen3-14b':
                analysis_eval.prompting_analysis_non_thinking(infra_model, infra_api=infra_api)
            else:
                raise ValueError('This model has no non-thinking mode.')
        else:
            analysis_eval.prompting_analysis(infra_model, infra_api=infra_api)
    elif args.analysis == 'agent':
        print(f'start analysis: agent')
        if args.non_thinking:
            if model == 'qwen3-14b':
                analysis_eval.agent_analysis_non_thinking(infra_model, infra_api=infra_api)
            else:
                raise ValueError('This model has no non-thinking mode.')
        else:
            analysis_eval.agent_analysis(infra_model, infra_api=infra_api)
    elif args.analysis == 'translate':
        print(f'start analysis: translate')
        analysis_eval.translate_analysis(infra_model, infra_api=infra_api)


if __name__ == '__main__':
    main()