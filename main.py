import os 
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed

from solution_code_gen import solve_task


def fmt_grid(inp):
    res = ''
    for row in inp:
        res += ''.join(map(str, row)) + '\n'
    return res


def validate_answers(predictions, targets):
    res = []
    for i in range(len(targets)):
        print("=" * 15)
        print(f"Correct: \n{fmt_grid(targets[i])}")
        print(f"Att1   : \n{fmt_grid(predictions[i]['attempt_1'])}")
        print(f"Att2   : \n{fmt_grid(predictions[i]['attempt_2'])}")
        
        if (predictions[i]['attempt_1'] == targets[i]) or (predictions[i]['attempt_2'] == targets[i]):
            res.append(1)
            print("CORRECT!!!!!!")
        else:
            res.append(0)
    return sum(res) / len(res)
    
    
def run_eval():
    res = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for i, task_filename in tqdm(enumerate(os.listdir('CCM_project/ARC-AGI/data/evaluation'))):
            if i < 0:
                continue
            with open('CCM_project/ARC-AGI/data/evaluation/' + task_filename, 'r') as f:
                task_data = json.load(f)
                correct_answers = [test_task['output'] for test_task in task_data['test']]
                answers = solve_task({
                    'train': task_data['train'],
                    'test_inputs': [test_task['input'] for test_task in task_data['test']],
                    'task_name': task_filename
                }, executor)
                # assert len(answers) == num_answers, f"You are outputing {len(answers)} answers, it should be {num_answers}"
                res.append(validate_answers(answers, correct_answers))
                print(f"Correct {sum(res)} / Tasks {len(res)}")
    res = np.array(res)
    print(f"Correct answers: {res.sum()}")
    print(f"Accuracy: {res.mean() * 100:.2f}%")
            

if __name__ == "__main__":
    run_eval()