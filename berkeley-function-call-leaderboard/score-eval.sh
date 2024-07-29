#!/bin/bash

cd eval_checker
python ./eval_runner.py --test-category simple multiple_function parallel_function parallel_multiple_function
