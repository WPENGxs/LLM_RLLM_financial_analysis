from openai import OpenAI
import copy

# openai
client_gpt = OpenAI(api_key="API KEY", base_url="https://api.openai.com/v1")

# deepseek
client_deepseek = OpenAI(api_key="API KEY", base_url="https://api.deepseek.com/v1")

# deepinfra
client_deepinfra = OpenAI(api_key="API KEY", base_url="https://api.deepinfra.com/v1/openai")

# aliyun
client_aliyun = OpenAI(api_key="API KEY", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

BREAK_TIMES_LIMIT = 1

class model():
    def __init__(self, model):
        self.model = model

    def gpt_generator(self, text, history=[]):
        tmp_history = copy.deepcopy(history)
        if tmp_history == []:
            tmp_history = [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": text},
            ]
        else:
            tmp_history.append({"role": "user", "content": text})
        loop_times = 0
        while True:
            if loop_times > BREAK_TIMES_LIMIT:
                message = 'BREAK_TIMES_LIMIT'
                break
            try:
                response = client_gpt.chat.completions.create(
                model=self.model,
                messages=tmp_history,
                stream=False)
                message = response.choices[0].message.content
            except Exception:
                message = 'Error'
            if message != 'Error':
                break
            loop_times += 1
        tmp_history.append({"role": "system", "content": message})
        return message, tmp_history

    def deepseek_generator(self, text, history=[]):
        tmp_history = copy.deepcopy(history)
        if tmp_history == []:
            tmp_history = [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": text},
            ]
        else:
            tmp_history.append({"role": "user", "content": text})
        loop_times = 0
        while True:
            if loop_times > BREAK_TIMES_LIMIT:
                message = 'BREAK_TIMES_LIMIT'
                break
            try:
                response = client_deepseek.chat.completions.create(
                model=self.model,
                messages=tmp_history,
                stream=False)
                message = response.choices[0].message.content
                if self.model == 'deepseek-reasoner':
                    reasoning_content = response.choices[0].message.reasoning_content
            except Exception:
                message = 'Error'
            if message != 'Error':
                break
            loop_times += 1
        tmp_history.append({"role": "assistant", "content": message})
        return message, tmp_history

    def deepinfra_generator(self, text, history=[]):
        tmp_history = copy.deepcopy(history)
        if tmp_history == []:
            tmp_history = [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": text},
            ]
        else:
            tmp_history.append({"role": "user", "content": text})
        loop_times = 0
        while True:
            if loop_times > BREAK_TIMES_LIMIT:
                message = 'BREAK_TIMES_LIMIT'
                break
            try:
                response = client_deepinfra.chat.completions.create(
                model=self.model,
                messages=tmp_history,
                stream=False)
                message = response.choices[0].message.content
            except Exception as e:
                # print(e)
                message = 'Error'
            if message != 'Error':
                break
            loop_times += 1
        tmp_history.append({"role": "system", "content": message})
        return message, tmp_history
    
    def aliyun_generator(self, text, history=[], input_extra_body={"enable_thinking": True}):
        tmp_history = copy.deepcopy(history)
        if tmp_history == []:
            tmp_history = [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": text},
            ]
        else:
            tmp_history.append({"role": "user", "content": text})
        loop_times = 0
        while True:
            if loop_times > BREAK_TIMES_LIMIT:
                message_ = 'BREAK_TIMES_LIMIT'
                break
            try:
                if input_extra_body["enable_thinking"]:
                    input_stream = True
                else:
                    input_stream = False
                response = client_aliyun.chat.completions.create(
                model=self.model,
                messages=tmp_history,
                extra_body=input_extra_body,
                stream=input_stream)
                if input_stream:
                    message_ = ""
                    message_ += "<think>\n"
                    thinking_end = False
                    for chunk in response:
                        if chunk.choices:
                            # print(chunk.choices[0].delta)
                            if chunk.choices[0].delta.reasoning_content != None:
                                message_ += str(chunk.choices[0].delta.reasoning_content)
                            if chunk.choices[0].delta.content != None:
                                if thinking_end == False:
                                    message_ += "\n</think>\n\n"
                                    thinking_end = True
                                message_ += str(chunk.choices[0].delta.content)
                else:
                    message_ = response.choices[0].message.content
            except Exception as e:
                message_ = 'Error'
                # print(e)
            if message_ != 'Error':
                break
            loop_times += 1
        tmp_history.append({"role": "system", "content": message_})
        return message_, tmp_history