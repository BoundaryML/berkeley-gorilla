

models=(
    # "ollama-mistral-BAML"
    "ollama-llama3.1"
    "ollama-llama3.1-BAML"
    "ollama-llama3.1-FC"
    # "gpt-4o-mini-2024-07-18"
    # "gpt-4o-mini-2024-07-18-FC"
    # "gpt-4o-mini-2024-07-18-BAML"
    # "gpt-4o-2024-05-13"
    # "claude-3-5-sonnet-20240620"
    # "gpt-4o-2024-05-13-FC"
    # "claude-3-5-sonnet-20240620-FC"
    # "claude-3-5-sonnet-20240620-BAML"
    # "gpt-4o-2024-05-13-BAML"
    # "gpt-3.5-turbo-0125"
    # "claude-3-haiku-20240307"
    # "gpt-3.5-turbo-0125-FC"
    # "claude-3-haiku-20240307-FC"
    # "gpt-3.5-turbo-0125-BAML"
    # "claude-3-haiku-20240307-BAML"
)

for model in "${models[@]}"; do
    python openfunctions_evaluation.py --model "$model" --max-parallel 5 --test-category ast
done
