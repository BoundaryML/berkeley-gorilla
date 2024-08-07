from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import (
    convert_to_tool,
    convert_to_function_call,
    augment_prompt_by_languge,
    language_specific_pre_processing,
    ast_parse,
)
from model_handler.constant import (
    GORILLA_TO_OPENAPI,
    GORILLA_TO_PYTHON,
    USER_PROMPT_FOR_CHAT_MODEL,
    SYSTEM_PROMPT_FOR_CHAT_MODEL,
)
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
import os, time, json


class OpenAIHandler(BaseHandler):
    def __init__(self, model_name: str, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        if "ollama-" in model_name:
            super().__init__(model_name, temperature, top_p, max_tokens)
            self.model_style = ModelStyle.Mistral
            self.client = OpenAI(api_key="placeholder", base_url = 'http://localhost:11434/v1')
            self.async_client = AsyncOpenAI(api_key="placeholder", base_url = 'http://localhost:11434/v1')
        else:
            super().__init__(model_name, temperature, top_p, max_tokens)
            self.model_style = ModelStyle.OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _build_request(self, prompt,functions,test_category):
        model_name = self.model_name
        if model_name.startswith("ollama-"):
            model_name = model_name.removeprefix("ollama-")

        if "FC" not in model_name:
            prompt = augment_prompt_by_languge(prompt,test_category)
            functions = language_specific_pre_processing(functions,test_category)
            message = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_FOR_CHAT_MODEL,
                },
                {
                    "role": "user",
                    "content": USER_PROMPT_FOR_CHAT_MODEL.format(
                        user_prompt=prompt, functions=str(functions)
                    ),
                },
            ]
            return dict(
                messages=message,
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            ), { "messages": message }
        else:
            prompt = augment_prompt_by_languge(prompt, test_category)
            functions = language_specific_pre_processing(functions, test_category)
            if type(functions) is not list:
                functions = [functions]
            message = [{"role": "user", "content": prompt}]
            oai_tool = convert_to_tool(
                functions, GORILLA_TO_OPENAPI, self.model_style, test_category, '-strict' in model_name
            )
            if len(oai_tool) > 0:
                return dict(
                    messages=message,
                    model=model_name.replace("-FC-strict", "").replace("-FC", ""),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    tools=oai_tool,
                    parallel_tool_calls='parallel' in test_category
                ), { "messages": message, "tools": oai_tool }
            else:
                return dict(
                    messages=message,
                    model=model_name.replace("-FC-strict", "").replace("-FC", ""),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                ), { "messages": message }
            
    def _handle_response(self, response: ChatCompletion, latency: float, prompt):
        if "FC" not in self.model_name:
            result = response.choices[0].message.content
        else:
            try:
                result = [
                    {func_call.function.name: func_call.function.arguments}
                    for func_call in response.choices[0].message.tool_calls
                ]
            except:
                result = response.choices[0].message.content
        metadata = {}
        metadata["input_tokens"] = response.usage.prompt_tokens
        metadata["output_tokens"] = response.usage.completion_tokens
        metadata["latency"] = latency
        metadata["prompt"] = prompt
        return result, metadata

    def inference(self, prompt,functions,test_category):
        params, prompt = self._build_request(prompt,functions,test_category)
        start_time = time.time()
        try:
            response: ChatCompletion = self.client.chat.completions.create(**params)
        except Exception as e:
            latency = time.time() - start_time
            return f"Error: {e}", {"input_tokens": 0, "output_tokens": 0, "latency": latency, "prompt": prompt, "error": str(e)}
        latency = time.time() - start_time
        return self._handle_response(response, latency, prompt)

    async def async_inference(self, prompt,functions,test_category):
        params, prompt = self._build_request(prompt,functions,test_category)
        start_time = time.time()
        try:
            response: ChatCompletion = await self.async_client.chat.completions.create(**params)
        except Exception as e:
            latency = time.time() - start_time
            return f"Error: {e}", {"input_tokens": 0, "output_tokens": 0, "latency": latency, "prompt": prompt, "error": str(e)}
        latency = time.time() - start_time        
        return self._handle_response(response, latency, prompt)

    def decode_ast(self,result,language="Python"):
        if "FC" not in self.model_name:
            decoded_output = ast_parse(result,language)
        else:
            if isinstance(result, str):
                raise ValueError("Result should be a list of function calls")
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                if "-strict" in self.model_name:
                    params = {
                        key: value
                        for key, value in params.items()
                        if value is not None
                    }
                decoded_output.append({name: params})
        return decoded_output
    
    def decode_execute(self,result):
        if "FC" not in self.model_name:
            decoded_output = ast_parse(result)
            execution_list = []
            for function_call in decoded_output:
                for key, value in function_call.items():
                    execution_list.append(
                        f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
                    )
            return execution_list
        else:
            if isinstance(result, str):
                raise ValueError("Result should be a list of function calls")
            function_call = convert_to_function_call(result)
            return function_call
