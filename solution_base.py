import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from prompts import direct_answer_prompt

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv('OPENAI_KEY'),
)
json_llm = llm.bind(response_format={"type": "json_object"})

class HumanDataGetter():
    def __init__(self):
        self.df = pd.read_csv('/Users/ksc/PycharmProjects/CCM/CCM_project/h-arc/data/data/clean_data.csv', low_memory=False)

    def get_by_task_id(self, task_name):
        candidates = self.df[self.df['task_name'] == task_name]
        candidates = candidates[candidates['solved'] == True]
        return candidates['last_written_solution'].unique().tolist()
    

def get_answer_by_gpt(task, test_input):
    # print(task['explanations'])
    ans = json_llm.invoke(direct_answer_prompt(task['train'], task['explanations'][:5], test_input))
    cnt = ans.content
    try:
        return eval(cnt)['output']
    except Exception:
        print("BAD OUTPUT")
        return []


hd = HumanDataGetter()    
def solve_task(task):
    task['explanations'] = hd.get_by_task_id(task['task_name'])
    
    ans = []
    for test_input in task['test_inputs']:
        ans.append({
            'attempt_1': get_answer_by_gpt(task, test_input),
            'attempt_2': get_answer_by_gpt(task, test_input)
        })
    return ans 