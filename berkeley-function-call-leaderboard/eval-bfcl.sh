#!/usr/bin/env bash

#export MODEL=mistral
export MODEL=gpt-3.5-turbo-0125
export MODEL=claude-3-haiku-20240307
export MODEL=gpt-4o-2024-05-13

cd eval_checker
poetry run python eval_runner.py --model "$MODEL" --test-category simple
poetry run python eval_runner.py --model "$MODEL" --test-category multiple_function
poetry run python eval_runner.py --model "$MODEL" --test-category parallel_function
poetry run python eval_runner.py --model "$MODEL" --test-category parallel_multiple_function
