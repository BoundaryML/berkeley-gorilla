#!/usr/bin/env bash

#export MODEL=mistral
export MODEL=gpt-3.5-turbo-0125

python openfunctions_evaluation.py --model "$MODEL" --test-category simple
python openfunctions_evaluation.py --model "$MODEL" --test-category multiple_function
python openfunctions_evaluation.py --model "$MODEL" --test-category parallel_function
python openfunctions_evaluation.py --model "$MODEL" --test-category parallel_multiple_function
