import argparse, json, os
import asyncio
from typing import Any, List, Tuple
from tqdm import asyncio as async_tqdm
from tqdm import tqdm
from model_handler.handler_map import handler_map
from model_handler.model_style import ModelStyle
from model_handler.handler import BaseHandler
from model_handler.constant import USE_COHERE_OPTIMIZATION
from eval_checker.eval_checker_constant import TEST_COLLECTION_MAPPING


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="gorilla-openfunctions-v2")
    # Refer to test_categories for supported categories.
    parser.add_argument("--test-category", type=str, default="all")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--timeout", default=60, type=int)
    parser.add_argument("--max-parallel", type=int, default=1)

    args = parser.parse_args()
    return args


TEST_FILE_MAPPING = {
    "executable_simple": "gorilla_openfunctions_v1_test_executable_simple.json",
    "executable_parallel_function": "gorilla_openfunctions_v1_test_executable_parallel_function.json",
    "executable_multiple_function": "gorilla_openfunctions_v1_test_executable_multiple_function.json",
    "executable_parallel_multiple_function": "gorilla_openfunctions_v1_test_executable_parallel_multiple_function.json",
    "simple": "gorilla_openfunctions_v1_test_simple.json",
    "relevance": "gorilla_openfunctions_v1_test_relevance.json",
    "parallel_function": "gorilla_openfunctions_v1_test_parallel_function.json",
    "multiple_function": "gorilla_openfunctions_v1_test_multiple_function.json",
    "parallel_multiple_function": "gorilla_openfunctions_v1_test_parallel_multiple_function.json",
    "java": "gorilla_openfunctions_v1_test_java.json",
    "javascript": "gorilla_openfunctions_v1_test_javascript.json",
    "rest": "gorilla_openfunctions_v1_test_rest.json",
    "sql": "gorilla_openfunctions_v1_test_sql.json",
}


def build_handler(model_name, temperature, top_p, max_tokens):
    handler = handler_map[model_name](model_name, temperature, top_p, max_tokens)
    return handler


def load_file(test_categories):
    test_to_run = []
    files_to_open = []

    if test_categories in TEST_COLLECTION_MAPPING:
        test_to_run = TEST_COLLECTION_MAPPING[test_categories]
        for test_name in test_to_run:
            files_to_open.append(TEST_FILE_MAPPING[test_name])
    else:
        test_to_run.append(test_categories)
        files_to_open.append(TEST_FILE_MAPPING[test_categories])

    return test_to_run, files_to_open


def find_test_cases(file_to_open: str, model: str) -> Tuple[List[Any], List[int]]:
    """
    Read the test cases from the file and return the test cases and the index of the test cases that have been already been tested.
    """
    print("Generating: " + file_to_open)

    test_cases = []
    with open("./data/" + file_to_open) as f:
        for line in f:
            test_cases.append(json.loads(line))
    num_existing_result = (
        []
    )  # if the result file already exists, skip the test cases that have been tested.
    if os.path.exists(
        "./result/"
        + model.replace("/", "_")
        + "/"
        + file_to_open.replace(".json", "_result.json")
    ):
        with open(
            "./result/"
            + model.replace("/", "_")
            + "/"
            + file_to_open.replace(".json", "_result.json")
        ) as f:
            for line in f:
                # Read the result file to find the number of test cases that have been tested.
                line = json.loads(line)
                index = line["idx"]
                num_existing_result.append(index)
    return test_cases, num_existing_result

def run_tests_sync(model: str, num_gpus: int, handler: BaseHandler, test_to_run, files_to_open):
    for test_category, file_to_open in zip(test_to_run, files_to_open):
        print("Generating: " + file_to_open)
        test_cases, existing_result = find_test_cases(file_to_open, model)

        if handler.model_style == ModelStyle.OSSMODEL:
            remaining_test_cases = [
                (i, test_cases[i])
                for i in range(len(test_cases))
                if i not in existing_result
            ]
            result = handler.inference(
                test_question=[case for (_, case) in remaining_test_cases],
                test_category=test_category,
                num_gpus=num_gpus,
            )
            for index, res in enumerate(result[0]):
                result_to_write = {
                    "idx": remaining_test_cases[index][0],
                    "result": res["text"],
                }
                handler.write(result_to_write, file_to_open)
        else:
            for index, test_case in enumerate(tqdm(test_cases)):
                if index in existing_result:
                    continue
                user_question, functions = test_case["question"], test_case["function"]
                if type(functions) is dict or type(functions) is str:
                    functions = [functions]
                result, metadata = handler.inference(
                    user_question, functions, test_category
                )
                result_to_write = {
                    "idx": index,
                    "result": result,
                    "input_token_count": metadata["input_tokens"],
                    "output_token_count": metadata["output_tokens"],
                    "latency": metadata["latency"],
                }
                handler.write(result_to_write, file_to_open)

async def run_tests_parallel(
    max_parallel: int,
    model: str,
    num_gpus: int,
    handler: BaseHandler,
    test_to_run,
    files_to_open,
):
    block = asyncio.Semaphore(max_parallel)
    for test_category, file_to_open in zip(test_to_run, files_to_open):
        test_cases, existing_result = find_test_cases(file_to_open, model)

        if handler.model_style == ModelStyle.OSSMODEL:
            remaining_test_cases = [
                (i, test_cases[i])
                for i in range(len(test_cases))
                if i not in existing_result
            ]
            result = await handler.async_inference(
                test_question=[case for (_, case) in remaining_test_cases],
                test_category=test_category,
                num_gpus=num_gpus,
            )
            for index, res in enumerate(result[0]):
                result_to_write = {
                    "idx": remaining_test_cases[index][0],
                    "result": res["text"],
                }
                handler.write(result_to_write, file_to_open)
        else:
            async def run_handler(index: int, test_case):
                user_question, functions = test_case["question"], test_case["function"]
                if type(functions) is dict or type(functions) is str:
                    functions = [functions]
                await block.acquire()
                try:
                    result, metadata = await handler.async_inference(
                        user_question, functions, test_category
                    )
                finally:
                    block.release()
                result_to_write = {
                    "idx": index,
                    "result": result,
                    "input_token_count": metadata["input_tokens"],
                    "output_token_count": metadata["output_tokens"],
                    "latency": metadata["latency"],
                }
                if "prompt" in metadata:
                    result_to_write["prompt"] = metadata["prompt"]
                handler.write(result_to_write, file_to_open)

            tasks = [
                run_handler(index, test_case)
                for index, test_case in enumerate(test_cases)
                if index not in existing_result
            ]

            if tasks:
                await async_tqdm.tqdm.gather(*tasks, total=len(tasks))

        # Now load the file and sort the results by index
        with open(
            "./result/"
            + model.replace("/", "_")
            + "/"
            + file_to_open.replace(".json", "_result.json"),
            mode="r+",
        ) as f:
            results = [json.loads(line) for line in f]
            results.sort(key=lambda x: x["idx"])
            f.seek(0)
            for result in results:
                f.write(json.dumps(result) + "\n")
            f.truncate()



if __name__ == "__main__":
    args = get_args()
    if USE_COHERE_OPTIMIZATION and "command-r-plus" in args.model:
        args.model = args.model + "-optimized"
    handler = build_handler(args.model, args.temperature, args.top_p, args.max_tokens)

    test_to_run, files_to_open = load_file(args.test_category)

    print("Running tests for model: ", args.model)

    if args.max_parallel > 1 and hasattr(handler, "async_inference"):
        asyncio.run(
            run_tests_parallel(
                args.max_parallel,
                args.model,
                args.num_gpus,
                handler,
                test_to_run,
                files_to_open,
            )
        )
    else:
        run_tests_sync(
                args.model,
                args.num_gpus, handler, test_to_run, files_to_open)
