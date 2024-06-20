from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator, Tuple
import json
import os

import pandas as pd
import matplotlib.pyplot as plt

# cost factor is $ per token
cost_factors = {
  'gpt-3.5-turbo-0125': {
    'input_tokens': 0.5 * 1e-6,
    'output_tokens': 1.5 * 1e-6,
  },
  'gpt-4o-2024-05-13': {
    'input_tokens': 5 * 1e-6,
    'output_tokens': 15 * 1e-6,
  },
  'claude-3-haiku-20240307': {
    'input_tokens': 0.25 * 1e-6,
    'output_tokens': 1.25 * 1e-6,
  },
}

test_categories = ['simple', 'multiple_function', 'parallel_function', 'parallel_multiple_function']

@dataclass
class TestSuite:
  model: str
  test_category: str

  # Total number of tests in the BFCL suite for this category
  test_count: int

  baml_run_count: int
  baml_skipped_count: int
  baml_pass_count: int
  baml_pass_rate: float

  bfcl_pass_count: int
  bfcl_pass_rate: float

  df: pd.DataFrame

  def summary(self):
    return {
      'model': self.model,
      'test_category': self.test_category,
      'test_count': self.test_count,
      'baml_run_count': self.baml_run_count,
      'baml_skipped_count': self.baml_skipped_count,
      'baml_pass_count': self.baml_pass_count,
      'bfcl_pass_count': self.bfcl_pass_count,
      'baml_pass_rate': self.baml_pass_rate,
      'bfcl_pass_rate': self.bfcl_pass_rate,
    }



def load_data() -> list[TestSuite]:
  """Transform the captured BAML event logs into usable data, and zip the BFCL data with it.

  Returns a list of:
    (model, test_category, df), e.g. ('gpt-3.5-turbo-0125', 'parallel_function', pd.DataFrame)
  """
  return list(yield_data())



def yield_data() -> Iterator[TestSuite]:
  """Transform the captured BAML event logs into usable data, and zip the BFCL data with it.

  Yields:
    (model, test_category, df), e.g. ('gpt-3.5-turbo-0125', 'parallel_function', pd.DataFrame)
  """
  for model in os.listdir('result'):

    if model not in cost_factors.keys():
      continue

    bfcl_result_path = f"../result/{model}"
    bfcl_data_by_category = {
      test_category: {i: json.loads(l) for i, l in enumerate(open(f"{bfcl_result_path}/gorilla_openfunctions_v1_test_{test_category}_result.json").readlines())}
      for test_category in test_categories
    }
    test_count_by_category = {
      test_category: len(test_data)
      for test_category, test_data
      in bfcl_data_by_category.items()
    }

    results_path = os.path.join('result', model)
    for test_category in test_categories:
      all_results_files = [f for f in os.listdir(results_path) if f.startswith(test_category)]
      if len(all_results_files) == 0:
        print(f"No results for {model}/{test_category}")
        continue
      latest_results_file = list(sorted(all_results_files))[-1]
      # if model == 'claude-3-haiku-20240307':
      #   latest_results_file = list(sorted(f for f in all_results_files if '19-09' in f))[-1]


      results_by_fn: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {})
      for l in open(os.path.join(results_path, latest_results_file), 'r').readlines():
        data = json.loads(l)
        results_by_fn[data['context']['event_chain'][0]['function_name']][data['event_type']] = data

      test_results_by_idx = {}
      for fn, results in results_by_fn.items():
        test_ordinal = int(fn.rsplit('_', 1)[1])
        idx = test_ordinal - 1

        test_result = {
          'idx': test_ordinal - 1,

          'baml.input_tokens': results['func_llm']['metadata'][0]['output']['metadata']['prompt_tokens'],
          'baml.output_tokens': results['func_llm']['metadata'][0]['output']['metadata']['output_tokens'],
          'baml.raw_output': results['func_llm']['metadata'][0]['output']['raw_text'],
          'baml.input': results['func_llm']['metadata'][0]['input']['prompt']['template'],
          'baml.latency_ms': results['func_llm']['context']['latency_ms'],
          'baml.outcome': results['func_code']['io']['output']['value'],

          'bfcl.input_tokens': bfcl_data_by_category[test_category][idx]['input_token_count'],
          'bfcl.output_tokens': bfcl_data_by_category[test_category][idx]['output_token_count'],
          'bfcl.latency_ms': bfcl_data_by_category[test_category][idx]['latency'] * 1000,
        }
        test_result['baml.output_tokens_per_sec'] = test_result['baml.output_tokens'] / test_result['baml.latency_ms'] * 1000
        test_result['bfcl.output_tokens_per_sec'] = test_result['bfcl.output_tokens'] / test_result['bfcl.latency_ms'] * 1000

        test_result['baml.total_tokens'] = test_result['baml.input_tokens'] + test_result['baml.output_tokens']
        test_result['bfcl.total_tokens'] = test_result['bfcl.input_tokens'] + test_result['bfcl.output_tokens']

        test_result['baml.cost'] = test_result['baml.input_tokens'] * cost_factors[model]['input_tokens'] + test_result['baml.output_tokens'] * cost_factors[model]['output_tokens']
        test_result['bfcl.cost'] = test_result['bfcl.input_tokens'] * cost_factors[model]['input_tokens'] + test_result['bfcl.output_tokens'] * cost_factors[model]['output_tokens']
        test_result['diff'] = {
          'input_token_diff': test_result['baml.input_tokens'] - test_result['bfcl.input_tokens'],
          'output_token_diff': test_result['baml.output_tokens'] - test_result['bfcl.output_tokens'],
          'latency_ms_diff': test_result['baml.latency_ms'] - test_result['bfcl.latency_ms'],
        }
        test_results_by_idx[test_result['idx']] = test_result

      test_results = [r for _, r in sorted(test_results_by_idx.items())]
      df = pd.DataFrame(test_results)

      bfcl_score = json.loads(open(f"../score/{model}/{test_category}_score.json", 'r').readlines()[0])

      test_count= bfcl_score['total_count']
      baml_run_count= len(test_results)
      baml_skipped_count = test_count - baml_run_count
      baml_pass_count = len([r for r in test_results if r['baml.outcome'] == "\"success\""])

      yield TestSuite(
        model=model,
        test_category=test_category,
        test_count=test_count,
        baml_run_count=baml_run_count,
        baml_skipped_count=baml_skipped_count,
        baml_pass_count=baml_pass_count,
        baml_pass_rate=baml_pass_count / test_count,
        bfcl_pass_count=bfcl_score['correct_count'],
        bfcl_pass_rate=bfcl_score['accuracy'],
        df=df,
      )

def plt_show():
  for t in load_data():
    metric = 'input_tokens'
    metric = 'output_tokens'
    metric = 'latency_ms'
    metric = 'cost'
    xlabel = f'baml.{metric}'
    ylabel = f'bfcl.{metric}'
    #f = plt.figure(f'{test_category}/{metric}')
    plt.scatter(t.df[xlabel], t.df[ylabel], marker='o')  # 'o' for circle markers
    #plt.plot([0, 0], [1000, 1000], 'r--', label='y = x Line')  # Plotting the y=x line

    plt.title(metric)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

    #for t in test_results: print(json.dumps(t['diff'], indent=2))
    print(json.dumps(t.summary(), indent=2))

  plt.show()


if __name__ == '__main__':
  plt_show()