from dataclasses import asdict, dataclass
from typing import Any, Tuple
import traceback
import json
import os
import shutil

from jinja2 import Environment, FileSystemLoader

# Constants
test_dir_path = "/Users/sam/repos/Berkeley-Function-Calling-Leaderboard/"
test_categories = ["simple", "multiple_function", "parallel_function", "parallel_multiple_function"]

# Function to prepend slashes to each line
def prepend_slashes(text: str):
    # Split the input text into lines and prepend a slash to each line
    return "\n".join("// " + line for line in text.splitlines())


env = Environment(
    loader=FileSystemLoader("."),
    autoescape=False,
)

env.filters["prepend_slashes"] = prepend_slashes

# llm_client = "Mistral"
llm_client = "GPT35Turbo"


def get_datasets():
  for test_category in test_categories:
    file_path = os.path.join(test_dir_path, f"gorilla_openfunctions_v1_test_{test_category}.json")
    # Open and read the JSON file
    with open(file_path, "r") as file:
        for lineno, line in enumerate(file.readlines(), start=1):
            if len(line.strip()) == 0:
                continue
            data = json.loads(line)
            yield (test_category, lineno, data)


def sanitize(fn_name: str) -> str:
    return fn_name.replace(".", "__").replace("-", "__")


def field_type_to_baml_type(prefix: str, field_type: dict[str, Any], enums: list["EnumType"], classes: list["ClassType"]) -> str:
    if field_type["type"] in ["string", "String"] and "enum" in field_type:
        enum_name = f"Enum_{prefix}_Enum{len(enums)}"
        enums.append(EnumType(name=enum_name, enum_values=[(f"Value{i}", v) for i, v in enumerate(field_type["enum"])]))
        return enum_name
    match field_type["type"]:
        case "String":
            return "string"
        case "string":
            return "string"
        case "integer":
            return "int"
        case "float":
            return "float"
        case "boolean":
            return "bool"
        case "array":
            array_type = field_type_to_baml_type(prefix, field_type["items"], enums, classes)
            return f"{array_type}[]"
        case "tuple":
            array_type = field_type_to_baml_type(prefix, field_type["items"], enums, classes)
            return f"{array_type}[]"
        case "dict":
            if 'properties' not in field_type:
                raise SyntaxError("skip: BAML does not support map types")
            fields = [
                OutputField(
                    field_name=field_name,
                    field_type="{}{}".format(
                        field_type_to_baml_type(prefix, field_details, enums, classes),
                        "" if field_name in field_type.get("required", []) else "?",
                    ),
                    description=field_details.get("description", ""),
                    default=field_details.get("default", ""),
                ) for field_name, field_details in field_type["properties"].items()
            ]
            class_name = f"Class_{prefix}_Class{len(classes)}"
            classes.append(ClassType(name=class_name, fields=fields))
            return class_name
        case _:
            raise SyntaxError(f"skip: no BAML type corresponds to {field_type}")


@dataclass
class LlmCall:
    filename: str
    lineno: int
    original_data: str
    fn_name: str
    fn_calls: list["FunctionCall"]
    output_type: str
    prompt: str
    llm_client: str
    enums: list["EnumType"]
    classes: list["ClassType"]


@dataclass
class EnumType:
    name: str

    # tuple of (enum_id, enum alias)
    enum_values: list[Tuple[str, str]]

@dataclass
class ClassType:
    name: str
    fields: list["OutputField"]


@dataclass
class FunctionCall:
    fn_id: str
    description: str
    output_type: str
    output_fields: list["OutputField"]


@dataclass
class OutputField:
    field_name: str
    field_type: str
    description: str
    default: str


def main():
    template = env.get_template("template-function-call.baml.j2")

    for f in os.listdir("baml_src"):
        if f.startswith("generated-"):
            shutil.rmtree(f"baml_src/{f}")

    all_render_args: dict[str, list[LlmCall]] = {}
    for filename, lineno, data in get_datasets():
        test_category = filename.replace("gorilla_openfunctions_v1_test_", "").replace(
            ".json", ""
        )
        fn_name = f"Fn_{test_category}_{lineno}"
        original_data = json.dumps(data, indent=2)
        # print(test_category)
        try:

            funcs = data["function"]
            if not isinstance(funcs, list):
                funcs = [funcs]
            fn_calls: list[FunctionCall] = []
            enums: list[EnumType] = []
            classes: list[ClassType] = []
            for i, fn in enumerate(funcs):
                # fn_name = f"Fn_{test_category}_{lineno}_{sanitize(data["function"]["name"])}"

                output_type = f"{fn_name}_Output{i}"
                output_fields: list[OutputField] = []

                if fn["parameters"]["type"] != "dict":
                    raise SyntaxError("skip: Function parameters are not of type dict")

                field_type = None
                for k, v in fn["parameters"]["properties"].items():
                    field_type = field_type_to_baml_type(fn_name, v, enums, classes)
                    if k not in fn["parameters"]["required"]:
                        if not field_type.endswith("[]"):
                            field_type = f"{field_type}?"
                    if k in ["_class", "_from", "class"]:
                        raise SyntaxError(f"skip: {k} is an invalid BAML property name")

                    output_fields.append(
                        OutputField(
                            field_name=k,
                            field_type=field_type,
                            description=v.get("description", ""),
                            default=v.get("default", ""),
                        )
                    )

                fn_calls.append(
                    FunctionCall(
                        fn_id=fn["name"],
                        description=fn["description"],
                        output_type=output_type,
                        output_fields=output_fields,
                    )
                )

            output_type="|".join([f.output_type for f in fn_calls])
            if "parallel" in test_category:
                output_type = f"({output_type})[]"

            render_args = LlmCall(
                filename=filename,
                lineno=lineno,
                original_data=original_data,
                fn_name=fn_name,
                fn_calls=fn_calls,
                enums=enums,
                classes=classes,
                output_type=output_type,
                prompt=data["question"],
                llm_client=llm_client,
            )
            all_render_args.setdefault(test_category, []).append(render_args)
            rendered = template.render(**asdict(render_args))

            generated_baml = f"baml_src/generated-{test_category}/{fn_name}.baml"
            os.makedirs(os.path.dirname(generated_baml), exist_ok=True)
            open(generated_baml, "w").write(rendered)

        except Exception as e:
            skip_reason = f"skipped {filename}:{lineno} due to {e}"
            print(skip_reason)
            skip_reason = f"skipped {filename}:{lineno} due to:\n\n{traceback.format_exc()}"

            generated_baml = f"baml_src/generated-{test_category}/{fn_name}.baml"
            os.makedirs(os.path.dirname(generated_baml), exist_ok=True)
            open(generated_baml, "w").write(f"{prepend_slashes(skip_reason)}\n\n{prepend_slashes(original_data)}")

    for test_category, render_arg_list in all_render_args.items():
        template = env.get_template("template-test.py.j2")
        rendered = template.render(
            tests=render_arg_list, test_category=test_category, llm_client=llm_client
        )
        generated_test = f"generated/test_{test_category}.py"
        os.makedirs(os.path.dirname(generated_test), exist_ok=True)
        open(generated_test, "w").write(rendered)


if __name__ == "__main__":
    main()
