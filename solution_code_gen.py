import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from solution_base import HumanDataGetter
from prompts import code_gen_prompt, hypothesis_synth_prompt, format_code_run_results
from langchain_community.chat_models import ChatOpenAI
import os
from typing import Optional
from multiprocessing import Process, Queue

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
    max_tokens=1000
)

MAX_FEEDBACK_ATTEMPTS = 10
MAX_PROGRAMS = 8
EVAL_TIMEOUT = 40  # seconds

from eval_code import eval_code  # Import after defining llm and classes above


def run_eval_in_subprocess(response_code, train_cases, test_inputs):
    # Run eval_code in a separate process to prevent infinite loops from blocking
    q = Queue()
    p = Process(target=eval_code, args=(response_code, train_cases, test_inputs, q))
    p.start()
    p.join(EVAL_TIMEOUT)
    if p.is_alive():
        print("Eval process timed out. Terminating...")
        p.terminate()
        p.join()
        return {"error": "Evaluation timed out due to potential infinite loop."}
    # If process finished, retrieve the result
    if not q.empty():
        return q.get()
    else:
        return {"error": "No result returned from evaluation."}


def process_hypothesis(hyp, task, test_inputs):
    prompt = code_gen_prompt(task['train'], hyp)
    for _ in range(MAX_FEEDBACK_ATTEMPTS):
        response = llm.invoke(prompt).content
        code_res = run_eval_in_subprocess(response, task['train'], test_inputs)

        if 'test_answers' in code_res:
            return code_res['test_answers']
        
        res_promt = format_code_run_results(code_res)
        prompt = [*prompt, ('assistant', response), ('user', res_promt)]
    return None


def code_gen_pipeline(task, test_inputs, executor):
    #H = [max(task['explanations'], key=len) for _ in range(MAX_PROGRAMS)]
    synthesized_prompt=hypothesis_synth_prompt(task['train'],task['explanations'])
    H = llm.invoke(synthesized_prompt)
    possible_hyp = [H for _ in range(MAX_PROGRAMS)]
    results = []

    future_to_hyp = [
        executor.submit(process_hypothesis, hyp, task, test_inputs) for hyp in possible_hyp
    ]
    for future in as_completed(future_to_hyp):
        try:
            result = future.result(timeout=40)
            if result is not None:
                return result
        except TimeoutError:
            print("A hypothesis timed out and was skipped.")
    return [[] for i in range(len(test_inputs))]    


hd = HumanDataGetter()    
def solve_task(task, executor):
    task['explanations'] = hd.get_by_task_id(task['task_name'])
    
    ans = []
    out = code_gen_pipeline(task, task['test_inputs'], executor)
    for el in out:
        ans.append({
            'attempt_1': el,
            'attempt_2': el
        })
    return ans
