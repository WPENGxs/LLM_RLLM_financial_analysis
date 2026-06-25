<h2 align="center">What Factors Affect LLMs and RLLMs in Financial Question Answering?</h2>

<p align="center">
  <b>
  [<a href="https://arxiv.org/abs/2507.08339">Arxiv</a>]
  </b>
  <b>
  [<a href="https://aclanthology.org/2026.findings-acl.752/">ACL 2026 Findings</a>]
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

> As the list of models provided by different API vendors is subject to change, some models may not be accessible. To run our code, you may switch to an API from a different vendor or deploy the models locally.

```bash
python main.py --analysis ANALYSIS --model MODEL [--non_thinking]
ANALYSIS = ['prompting', 'agent', 'multilingual', 'translate']
MODEL = ['llama-3.1-8b', 'gpt-4o-mini', 'gemini-1.5-flash', 'qwen-2.5-32b', 'deepseek-v3', 'deepseek-r1-dis', 'gpt-oss-20b', 'qwen3-14b', 'o4-mini']
--non_thinking = Qwen-3 non-thinking mode
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

For DeepInfra, you can find detailed settings under Model Card > API > OpenAI Chat Completions > Input fields.

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

### GPT-OSS-20B
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
@inproceedings{wang-etal-2026-factors,
    title = "What Factors Affect {LLM}s and {RLLM}s in Financial Question Answering?",
    author = "Wang, Peng  and
      Hu, Xuesi  and
      Wu, Jiageng  and
      Zou, Yuntao  and
      Zhang, Qiancheng  and
      Li, Dagang",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.752/",
    pages = "15315--15327",
    ISBN = "979-8-89176-395-1",
    abstract = "Recently, large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and four RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. Additionally, we discuss strategies for enhancing the performance of LLMs and RLLMs in financial question answering, which may serve as a inspiration for future improvements. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering."
}
```

## Contact
If you have any questions or suggestions, please create Github issues here or email [Peng Wang](mailto:wpengxss@gmail.com).
