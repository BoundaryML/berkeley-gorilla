from model_handler.model_style import ModelStyle
import json, os



class BaseHandler:
    model_name: str
    model_style: ModelStyle

    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def inference(self, prompt, functions, test_category):
        # This method is used to retrive model response for each model.
        pass

    def decode_ast(self, result, language="Python"):
        # This method takes raw model output and convert it to standard AST checker input.
        pass

    def decode_execute(self, result):
        # This method takes raw model output and convert it to standard execute checker input.
        pass

    def result_directory(self):
        model_name_dir = self.model_name.replace("/", "_")
        return f"./result/{model_name_dir}"

    def result_file(self, test_category):
        return f"{self.result_directory()}/gorilla_openfunctions_v1_test_{test_category}_result.json"

    def write(self, result):
        os.makedirs(self.result_directory(), exist_ok=True)
        
        if type(result) is dict:
            result = [result]
            
        for entry in result:
            test_category = entry["id"].rsplit("_", 1)[0]
            file_to_write = self.result_file(test_category)
            
            with open(file_to_write, "a+") as f:
                f.write(json.dumps(entry) + "\n")

    def write_sorted(self, test_category):
        # Load the files
        with open(self.result_file(test_category), "r+") as f:
            data = f.readlines()
            # Sort the data by id
            data = sorted(data, key=lambda x: json.loads(x)["id"])
            # Write the sorted data back to the file
            f.seek(0)
            f.writelines(data)
            f.truncate()
