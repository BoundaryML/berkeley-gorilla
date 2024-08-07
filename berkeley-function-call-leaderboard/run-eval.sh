

base_models=(
    "gpt-4o-mini-2024-07-18"
    # "ollama-llama3.1"
    "gpt-4o-2024-05-13"
    "claude-3-5-sonnet-20240620"
    "gpt-3.5-turbo-0125"
    "claude-3-haiku-20240307"
)

types=(
    ""
    "-BAML"
    "-FC"
    "-FC-strict"
)

for type in "${types[@]}"; do
    for base_model in "${base_models[@]}"; do
        model="${base_model}${type}"
        python openfunctions_evaluation.py --model "$model" --max-parallel 50 --temperature 0 --test-category simple multiple_function parallel_function parallel_multiple_function executable_simple executable_multiple_function executable_parallel_function executable_parallel_multiple_function relevance
    done
done
