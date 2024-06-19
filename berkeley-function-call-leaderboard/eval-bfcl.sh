#!/usr/bin/env bash

#export MODEL=mistral
export MODEL=gpt-3.5-turbo-0125

cd eval_checker
python eval_runner.py --model "$MODEL" --test-category simple
python eval_runner.py --model "$MODEL" --test-category multiple_function
python eval_runner.py --model "$MODEL" --test-category parallel_function
python eval_runner.py --model "$MODEL" --test-category parallel_multiple_function
