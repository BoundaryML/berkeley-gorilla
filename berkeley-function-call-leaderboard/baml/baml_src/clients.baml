generator my_client {
    output_type "python/pydantic"
    output_dir "../"
}

client<llm> GPT4 {
  provider baml-openai-chat
  options {
    model gpt-4
    api_key env.OPENAI_API_KEY
  }
} 

client<llm> GPT4o {
  provider baml-openai-chat
  options {
    model gpt-4o
    api_key env.OPENAI_API_KEY
  }
} 


client<llm> GPT4Turbo {
  provider baml-openai-chat
  options {
    model gpt-4-turbo
    api_key env.OPENAI_API_KEY
  }
} 

client<llm> GPT35Turbo {
  provider baml-openai-chat
  options {
    model gpt-3.5-turbo-0125
    api_key env.OPENAI_API_KEY
  }
}

client<llm> Llama2 {
  provider ollama
  options {
    model llama2
  }
}

client<llm> Mistral {
  provider ollama
  options {
    model mistral
  }
}

client<llm> GPT35Azure {
  provider azure-openai
  options {
    resource_name "west-us-azure-baml"
    deployment_id "gpt-35-turbo-default"
    // base_url "https://west-us-azure-baml.openai.azure.com/openai/deployments/gpt-35-turbo-default"
    api_version "2024-02-01"
    api_key env.AZURE_OPENAI_API_KEY
  }
}

client<llm> ClaudeHaiku {
  provider anthropic
  options {
    model claude-3-haiku-20240307
    api_key env.ANTHROPIC_API_KEY
    max_tokens 1000
  }
}