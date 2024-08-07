class Response {
    @@dynamic
}


template_string FunctionHeader(function_name: string, description: string) #"
// {{ description }}
"#

class Function {
    name string
    description string
}

function SimpleFunction(functions: Function[], query: string) -> Response {
    client GPT35Turbo
    prompt #"
        Pretend you are trying to fill out an api request to a function called '{{ functions[0].name }}'. The goal is {{ functions[0].description }}

        If a parameter isn't explicitly mentioned, don't assume any values unless its certain.

        {{ ctx.output_format }}

        {{ _.role('user') }}
        Question: {{ query }}
    "#
}

function ParallelFunction(functions: Function[], query: string) -> Response[] {
    client GPT35Turbo
    prompt #"
        Pretend you are trying to fill out multiple API calls to a function '{{ functions[0].name }}'. The goal is {{ functions[0].description }}

        If a parameter isn't explicitly mentioned, don't assume any values unless its certain.

        {{ ctx.output_format }}

        {{ _.role('user') }}
        Question: {{ query }}
    "#
}


function MultipleFunctions(functions: Function[], query: string) -> Response {
    client GPT35Turbo
    prompt #"
        Pretend you are trying to pick the best API to call given a question. Pick the best API, and fill out the parameters.

        If a parameter isn't explicitly mentioned, don't assume any values unless its certain.

        {{ ctx.output_format }}

        {{ _.role('user') }}
        Question: {{ query }}
    "#
}

function ParallelMultipleFunctions(functions: Function[], query: string) -> Response[] {
    client GPT35Turbo
    prompt #"
        Pretend you are trying to select all the relevant APIs to call given a question. Select all and fill out the parameters.

        If a parameter isn't explicitly mentioned, don't assume any values unless its certain.

        {{ ctx.output_format }}

        {{ _.role('user') }}
        Question: {{ query }}
    "#
}


function RelevanceFunction(functions: Function[], query: string) -> Response? {
    client GPT35Turbo
    prompt #"
        Pretend you are trying to fill out an api request to a function called '{{ functions[0].name }}'. The goal is {{ functions[0].description }}

        If a parameter isn't explicitly mentioned, don't assume any values unless its certain. If function '{{ functions[0].name }}' doesn't seem relevant, return null.

        {{ ctx.output_format }}

        {{ _.role('user') }}
        Question: {{ query }}
    "#
}

test TestName {
  functions [SimpleFunction]
  args {
    functions [
      { name "calculate_triangle_area", description "Calculate the area of a triangle given its base and height" }
    ]
    query #"
       Find the area of a triangle with a base of 10 units and height of 5 units.
    "#
  }
}
