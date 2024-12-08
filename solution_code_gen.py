import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed

from solution_base import HumanDataGetter
from prompts import code_gen_prompt
from eval_code import eval_code
from prompts import format_code_run_results
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os
from typing import Optional

load_dotenv()

class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        openai_api_key = os.getenv('OPENROUTER_API_KEY')
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)
        
llm = ChatOpenRouter(
    model_name="qwen/qwen-2.5-coder-32b-instruct",
    max_tokens = 1000
)

MAX_FEEDBACK_ATTEMPTS = 2
MAX_PROGRAMS = 8



def process_hypothesis(hyp, task, test_inputs):
    prompt = code_gen_prompt(task['train'], hyp)
    for _ in range(MAX_FEEDBACK_ATTEMPTS):
        response = llm.invoke(prompt).content
        code_res = eval_code(response, task['train'], test_inputs)
        
        if 'test_answers' in code_res:
            return code_res['test_answers']
        
        res_promt = format_code_run_results(code_res)
        prompt = [*prompt, ('assistant', response), ('user', res_promt)]
    return None


def code_gen_pipeline(task, test_inputs, executor):
    H = [max(task['explanations'], key=len) for _ in range(MAX_PROGRAMS)]
    possible_hyp = [H for _ in range(MAX_PROGRAMS)]
    
    results = []

    future_to_hyp = [
        executor.submit(process_hypothesis, hyp, task, test_inputs) for hyp in possible_hyp
    ]
    for future in as_completed(future_to_hyp):
        try:
            result = future.result(timeout=40)
            if result is not None:  # If a valid result is found, return it immediately
                return result
        except TimeoutError:
            print("A hypothesis timed out and was skipped.")
    return [[] for i in range(len(test_inputs))]    
    # print('\n----\n'.join(task['explanations']))
    # hypothesis = input()
    # hypothesis = ' | '.join(sorted(task['explanations'], key=len)[-5:])
    # hypothesis = " Double the size of the original grid. Create 4-square colored square on larger grid in the same spot where original colored square is. Use blue squares to fill in diagonally from bottom right and top left of any four-square to bottom right and top left of grid "
    # print(f"PICKED: {hypothesis}")
    


    
hd = HumanDataGetter()    
def solve_task(task):
    task['explanations'] = hd.get_by_task_id(task['task_name'])
    
    ans = []

    out = code_gen_pipeline(task, task['test_inputs'], executor)
    for el in out:
        ans.append({
                'attempt_1': el,
                'attempt_2': el
            })
    return ans 