import asyncio
import json
import time
from typing import TypedDict, List, Dict, Literal, Union

from model_handler.utils import convert_to_function_call
from model_handler.model_style import ModelStyle
from model_handler.handler import BaseHandler
from model_handler.baml_client import b
from .baml_client.type_builder import TypeBuilder, FieldType
from .baml_client.types import Response
from baml_py.baml_py import FunctionResult, ClientRegistry

class Function(TypedDict):
    name: str
    description: str
    parameters: "MapParameters"


class MapParameters(TypedDict):
    type: Literal["dict"]
    properties: Dict[str, Union["BaseParam", "ArrayParam", "MapParameters"]]
    required: List[str]

class BaseParam(TypedDict):
    type: Literal["string", "integer", "boolean", "float"]
    description: str

class ArrayParam(BaseParam):
    type: Literal["array", "tuple"]
    items: Union["BaseParam", "ArrayParam", "MapParameters"]
    description: str

def random_name():
    import random
    import string
    return "".join(random.choices(string.ascii_lowercase, k=10))

class UnsupportedType(Exception):
    def __init__(self, type: str):
        self.type = type
    
    def __str__(self):
        return f"Unsupported type: {self.type}"

def get_type_base(p: MapParameters | BaseParam | ArrayParam, tb: TypeBuilder) -> FieldType:
    match p['type']:
        case "array":
            return get_type_base(p['items'], tb).list()    
        case "tuple":
            return get_type_base(p['items'], tb).list()     
        case "dict":
            if "properties" not in p:
                return tb.map(tb.string(), tb.string())
            else:
                c = tb.add_class(random_name())
                for name, param in p['properties'].items():
                    prop = c.add_property(name, get_type(param, tb, name in p["required"] if "required" in p else False))
                    if "description" in param:
                        desc = param['description']
                        if 'default' in param and param['default']:
                            desc += f". Default to '{param['default']}'"
                        prop.description(desc)
                return c.type()
        case "string":
            # Possible for this to be an enum, so try that.
            if "enum" in p:
                enm = tb.add_enum(random_name())
                for value in p["enum"]:
                    enm.add_value(value)
                # We make all enums optional to enable smoother parsing
                return enm.type().optional()
            return tb.string()
        case "integer":
            return tb.int()
        case "boolean":
            return tb.bool()
        case "double":
            return tb.float()
        case "float":
            return tb.float()
        case "any":
            return tb.string()
        case other:
            # print(f"Unknown type: {other} - {p}")
            return tb.string()
    raise UnsupportedType(p['type'])

def get_type(type: str, tb: TypeBuilder, required: bool):
    base = get_type_base(type, tb)
    if required:
        return base
    else:
        return base.optional()

def update_metadata(metadata: Dict[str, Union[int, float]], fr: FunctionResult):    
    raw = json.loads(fr.internals())

    if "Success" in raw:
        response = raw["Success"]
        latency_s = response["latency"]["secs"]
        latency_ns = response["latency"]["nanos"]
        latency_ms = latency_s * 1000 + latency_ns / 1e6
        metadata["latency"] = latency_ms / 1000

        metadata["input_tokens"] = response["metadata"].get("prompt_tokens", 0)
        metadata["output_tokens"] = response["metadata"].get("output_tokens", 0)
        metadata["prompt"] = {
            "messages": response["prompt"]["Chat"],
            "parsed": response["content"]
        }
    elif "Error" in raw:
        print("ERROR", raw)
        response = raw["Error"]
        latency_s = response["latency"]["secs"]
        latency_ns = response["latency"]["nanos"]
        latency_ms = latency_s * 1000 + latency_ns / 1e6
        metadata["latency"] = latency_ms / 1000
        metadata["error"] = response["message"]
    elif "OtherFailure" in raw:
        metadata["error"] = raw["OtherFailure"]

    try:
        response = fr.parsed()
        return response
    except Exception as e:
        if "error" not in metadata:
            metadata["error"] = str(e)
    return None
    

def get_provider(model_name: str):
    if model_name.startswith("gpt-"):
        return "openai", dict(model=model_name)
    elif model_name.startswith("ollama-"):
        return "openai", dict(model=model_name.removeprefix("ollama-"), base_url="http://localhost:11434/v1")
    elif model_name.startswith("claude"):
        return "anthropic", dict(model=model_name)
    raise ValueError(f"Unknown model: {model_name}")



class BAMLHandler(BaseHandler):
    def __init__(self, model_name: str, temperature=0.7, top_p=1, max_tokens=1000):
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.cr = ClientRegistry()
        model_name = model_name.removesuffix("-BAML")
        provider, params = get_provider(model_name)
        if provider == "openai":
            self.model_style = ModelStyle.OpenAI
        else:
            self.model_style = ModelStyle.Anthropic_Prompt
        params["temperature"] = temperature
        params["top_p"] = top_p
        params["max_tokens"] = max_tokens
        self.cr.add_llm_client("Runner", provider, params)
        self.cr.set_primary("Runner")

    def inference(self, prompt, functions, test_category):
        return asyncio.run(self.async_inference(prompt, functions, test_category))

    async def async_inference(self, prompt, functions: List[Function], test_category):
        metadata = {
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": 0,
        }

        is_simple = test_category in ['relevance', 'java', 'javascript', 'simple', 'parallel_function', 'executable_simple', 'executable_parallel_function', 'rest']
        is_multiple = test_category in ['multiple_function', 'parallel_multiple_function', 'executable_multiple_function', 'executable_parallel_multiple_function']

        # This method is used to retrive model response for each model.
        if is_simple:
            assert len(functions) == 1
            function = functions[0]
            params = function['parameters']

            tb = TypeBuilder()
            try:
                # literal = tb.add_enum(random_name())
                # literal.add_value(function["name"])
                # tb.Response.add_property("function_name", literal.type().optional()).description(function["description"])
                if 'relevance' in test_category:
                    enm = tb.add_enum(random_name())
                    enm.add_value(function["name"])
                    tb.Response.add_property("function_name", enm.type()).description(function["description"])
                for name, param in params['properties'].items():
                    prop = tb.Response.add_property(name, get_type(param, tb, name in params['required']))
                    if "description" in param:
                        desc = param['description']
                        if 'default' in param and param['default']:
                            desc += f". Default to '{param['default']}'"
                        prop.description(desc)
                
                rt, ctx = b.z_unstable_runtime(), b.z_unstable_ctx_manager()
                raw = await rt.call_function(
                    "ParallelFunction" if 'parallel' in test_category else ("RelevanceFunction" if 'relevance' in test_category else "SimpleFunction"),
                    {
                        "functions": [{
                            "name": function["name"],
                            "description": function["description"],
                        }],
                        "query": prompt,
                    },
                    ctx.get(),
                    tb._tb,
                    self.cr
                )

                response = update_metadata(metadata, raw)
                if 'error' in metadata:
                    result = None
                else:
                    if test_category == 'relevance':
                        if response is None:
                            result = None
                        else:
                            response = Response.model_validate(response)
                            result = [
                                {
                                    function['name']: response.model_dump_json(exclude_none=True)
                                }
                            ]
                    elif test_category == 'simple':
                        response = Response.model_validate(response)
                        result = [
                            {
                                function['name']: response.model_dump_json(exclude_none=True)
                            }
                        ]
                    else:
                        response = [Response.model_validate(r) for r in response]
                        result = [
                            {
                                # BFCL uses "" to represent None
                                function['name']: Response.model_validate(r).model_dump_json(exclude_none=True)
                            }
                            for r in response
                        ]
            except UnsupportedType as e:
                metadata["error"] = str(e)
                result = None
            except Exception as e:
                metadata["error"] = str(e)
                result = None                     
        elif is_multiple:
            tb = TypeBuilder()
            try:
                response = []
                for function in functions:
                    cls = tb.add_class(function["name"])
                    params = function['parameters']
                    
                    e = tb.add_enum(random_name())
                    e.add_value(function["name"])
                    cls.add_property("function_name", e.type()).description(function["description"])

                    for name, param in params['properties'].items():
                        prop = cls.add_property(name, get_type(param, tb, name in params['required']))
                        if "description" in param:
                            desc = param['description']
                            if 'default' in param and param['default']:
                                desc += f". Default to '{param['default']}'"
                            prop.description(desc)
                    
                    response.append(cls.type())
                tb.Response.add_property("function", tb.union(response)).alias("selected")

                rt = b.z_unstable_runtime()
                ctx = b.z_unstable_ctx_manager()
                raw = await rt.call_function(
                    "ParallelMultipleFunctions" if 'parallel' in test_category else "MultipleFunctions",
                    {
                        "functions": [
                            {
                                "name": function["name"],
                                "description": function["description"],
                            } for function in functions
                        ],
                        "query": prompt,
                    },
                    ctx.get(),
                    tb._tb,
                    self.cr
                )
                response = update_metadata(metadata, raw)
                if 'error' in metadata:
                    result = None
                else:
                    if test_category == 'multiple_function':
                        response = Response.model_validate(response)                
                        result = [
                            {
                                response.function['function_name']: json.dumps(response.function)
                            }
                        ]
                    else:
                        response = [Response.model_validate(r) for r in response]
                        result = [
                            {
                                # BFCL uses "" to represent None
                                r.function['function_name']: json.dumps(r.function)
                            }
                            for r in response
                        ]
            except UnsupportedType as e:
                metadata["error"] = str(e)
                result = None
            except Exception as e:
                metadata["error"] = str(e)
                result = None     
        else:
            raise ValueError(f"Unknown test category: {test_category}")

        
        
        return result, metadata

    def decode_ast(self, result, language="Python"):
        if result is None:
            raise ValueError("Result should be a list of function calls")

        decoded_output = []
        for invoked_function in result:
            if invoked_function:
                name = list(invoked_function.keys())[0]
                params = {
                    key: value
                    for key, value in json.loads(invoked_function[name]).items()
                    if value is not None
                }
                if 'function_name' in params:
                    del params['function_name']
                if language == "Python":
                    pass
                else:
                    # all values of the json are casted to string for java and javascript
                    for key in params:    
                        params[key] = str(params[key])
                decoded_output.append({name: params})
        return decoded_output

    def decode_execute(self, result):
        if result is None:
            raise ValueError("Result should be a list of function calls")
        
        execution_list = []
        for invoked_function in result:
            if invoked_function:
                name = list(invoked_function.keys())[0]
                params = {
                    key: value
                    for key, value in json.loads(invoked_function[name]).items()
                    if value is not None
                }
                if 'function_name' in params:
                    del params['function_name']
                execution_list.append(f"{name}({','.join([f'{k}={repr(v)}' for k, v in params.items()])})")
        return execution_list

