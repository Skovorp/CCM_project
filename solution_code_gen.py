import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from solution_base import HumanDataGetter
from prompts import code_gen_prompt
from eval_code import eval_code
from prompts import format_code_run_results

load_dotenv()

llm = ChatOpenAI(
    # model="gpt-4o-mini-2024-07-18",
    model="gpt-4o-2024-08-06",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv('OPENAI_KEY'),
)

MAX_FEEDBACK_ATTEMPTS = 2
MAX_PROGRAMS = 16

# TODO: асинк
# если хоть сколько-то решит - победа



def code_gen_pipeline(task, test_inputs):
    # possible_hyp = [max(task['explanations'], key=len) for _ in range(MAX_PROGRAMS)]
    H = """1. Double the size of the original grid, creating a larger grid.
2. For each colored square in the original grid, create a 2x2 block of the same color in the corresponding position on the larger grid.
3. Use blue squares to form diagonal lines. Starting from the bottom-right corner of any 2x2 colored block, fill diagonally outward towards the bottom-right. Also from top-left corner of any 2x2 colored block, fill diagonally outward towards the top-left."""
    possible_hyp = [H for _ in range(MAX_PROGRAMS)]
    for hyp in possible_hyp:
        prompt = code_gen_prompt(task['train'], hyp)
        for _ in range(MAX_FEEDBACK_ATTEMPTS):
            response = llm.invoke(prompt).content
            code_res = eval_code(response, task['train'], test_inputs)
            
            if 'test_answers' in code_res:
                return code_res['test_answers']
            
            res_promt = format_code_run_results(code_res)
            prompt = [*prompt, ('assistant', response), ('user', res_promt)]
        
    # print('\n----\n'.join(task['explanations']))
    # hypothesis = input()
    # hypothesis = ' | '.join(sorted(task['explanations'], key=len)[-5:])
    # hypothesis = " Double the size of the original grid. Create 4-square colored square on larger grid in the same spot where original colored square is. Use blue squares to fill in diagonally from bottom right and top left of any four-square to bottom right and top left of grid "
    # print(f"PICKED: {hypothesis}")
    


    
hd = HumanDataGetter()    
def solve_task(task):
    task['explanations'] = hd.get_by_task_id(task['task_name'])
    
    ans = []

    out = code_gen_pipeline(task, task['test_inputs'])
    for el in out:
        ans.append({
                'attempt_1': el,
                'attempt_2': el
            })
    return ans 