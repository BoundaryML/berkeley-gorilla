#!/usr/bin/env bash

set -euo pipefail

export BOUNDARY_BASE_URL=http://localhost:8080
export BOUNDARY_SECRET=b-secret
export BOUNDARY_PROJECT_ID=b-project-id
# BAML_CLIENT is a 2-tuple: first entry controls BAML codegen, second controls report codegen
export BAML_CLIENT="GPT35Turbo:gpt-3.5-turbo-0125"
#export BAML_CLIENT="Mistral:mistral"
#export BAML_CLIENT="Llama3:llama3"
export REPORT_DATE=$(date '+%Y-%m-%d-%H-%M-%S')

poetry run python generate_code.py
poetry run baml-cli generate --from baml_src

poetry run python run_baml_event_logger.py &
export TRACING_SERVER_PID=$!
trap "echo 'Shutting down BAML event logger pid=$TRACING_SERVER_PID'; kill $TRACING_SERVER_PID" EXIT


export TEST_CATEGORY=test_simple
poetry run python -m generated_py.$TEST_CATEGORY
export TEST_CATEGORY=test_parallel_function
export TEST_CATEGORY=test_multiple_function
export TEST_CATEGORY=test_parallel_multiple_function