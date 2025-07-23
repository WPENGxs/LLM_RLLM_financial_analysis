<h2 align="center">What Factors Affect LLMs and RLLMs in Financial Question Answering?</h2>

<p align="center">
  <b>
  [<a href="https://arxiv.org/abs/2507.08339">Arxiv</a>]
  </b>
  <br/>
</p>

## Install
Install environment:
```bash
pip install openai tqdm scikit-learn pandas pyarrow tabulate
```
Prepare API Key in `model.py`:
```
    # openai
--> client_gpt = OpenAI(api_key="API KEY", base_url="https://api.openai.com/v1")

    # deepseek
--> client_deepseek = OpenAI(api_key="API KEY", base_url="https://api.deepseek.com/v1")

    # deepinfra
--> client_deepinfra = OpenAI(api_key="API KEY", base_url="https://api.deepinfra.com/v1/openai")

    # aliyun
--> client_aliyun = OpenAI(api_key="API KEY", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
```

## Run
```bash
python main.py --analysis ANALYSIS --model MODEL [--non_thinking]
ANALYSIS = ['prompting', 'agent', 'multilingual', 'translate']
MODEL = ['llama-3.1-8b', 'gpt-4o-mini', 'gemini-1.5-flash', 'qwen-2.5-32b', 'deepseek-v3', 'deepseek-r1-dis', 'qwen3-14b', 'o4-mini']
```

## Eval
First, get the evaluation results:
```bash
python evaluation.py --eval EVAL
EVAL = ['prompting', 'agent', 'multilingual', 'translate']
```

Then calculate the final performance:
```bash
python cal_acc.py
```

## Detailed Information
We use the default settings of the API platform. Here are the details of our experiments:

[[OpenAI](https://platform.openai.com/docs/api-reference/evals/object)] / 
[[Aliyun](https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api)] / 
[[DeepInfra](https://deepinfra.com/meta-llama/Meta-Llama-3.1-8B-Instruct/api)] / 
[[DeepSeek](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion)]

### Llama-3.1-8B-Instruct
```
temperature=1
top_p=1
```

### gpt-4o-mini
```
temperature=1
top_p=1
```

### Gemini-1.5-flash
```
temperature=1
top_p=1
```

### Qwen-2.5-32B
```
temperature=0.7
top_p=0.8
```

### DeepSeek-V3
```
temperature=1
top_p=1
```

### DeepSeek-R1-Distill-Qwen-32B
```
temperature=1
top_p=1
```

### Qwen-3-14B (thinking)
```
temperature=0.6
top_p=0.95
```

### Qwen-3-14B/8B/4B (non-thinking)
```
temperature=0.7
top_p=0.8
```

### O4-mini
```
temperature=1
top_p=1
```

## Reference
If you find this project useful for your research, please consider citing the following paper:
```
@article{wang2025factors,
  title={What Factors Affect LLMs and RLLMs in Financial Question Answering?},
  author={Wang, Peng and Hu, Xuesi and Wu, Jiageng and Zou, Yuntao and Zhang, Qiancheng and Li, Dagang},
  journal={arXiv preprint arXiv:2507.08339},
  year={2025}
}
```

## Contact
If you have any questions or suggestions, please create Github issues here or email [Peng Wang](mailto:wpengxss@gmail.com).
