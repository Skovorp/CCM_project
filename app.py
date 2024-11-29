import os
import json
import asyncio
import aiofiles
import openai
import pandas as pd
import time
from colors import get_colors
from langchain.chat_models import ChatOpenAI

openai.api_key = ""
semaphore = asyncio.Semaphore(5) #log after fine-tune 28-11

TRAINING_DATA_PATH = r"C:\Users\abhin\Desktop\larc_data\ARC-AGI\data\training"
EVALUATION_DATA_PATH = r"C:\Users\abhin\Desktop\larc_data\ARC-AGI\data\evaluation"
CLEAN_DATA_CSV = 'clean_data.csv'  

COLORS = get_colors()

def get_system_prompt(cfg):
    """Generates the system prompt for language generation."""
    system_prompt = 'You will receive a list of input-output pairs, where each input and output represents a grid of numbers forming a visual pattern. A SINGLE consistent pattern exists that transforms each input grid into its corresponding output grid.'

    if cfg.language_generation.add_meta_hint:
        system_prompt += '''
The pattern may involve various operations, such as counting or sorting objects (e.g., sorting by size), comparing numbers (e.g., determining the most frequent shape or symbol, identifying the largest object, or finding objects of the same size), or repeating a pattern a fixed number of times.

Additional relevant concepts include:
- Lines and rectangular shapes
- Symmetries, such as rotations and translations
- Shape upscaling or downscaling, and elastic distortions
- Containment relationships (e.g., inside or outside a perimeter)
- Drawing lines, connecting points, and orthogonal projections
- Copying or repeating objects

Please treat black cells as empty (background) cells.
'''
    system_prompt += '\nThe number in the input grid can be mapped to the following colors:'
    for idx, color in enumerate(COLORS):
        system_prompt += f" {idx}:{color['color_name']} (RGB: {color['rgb']});"

    system_prompt += '\nOutput the language description of the transformation.'

    if cfg.language_generation.example_tasks == 0:
        system_prompt += 'Your description should be in the format:\nDescribing the input grid:{text}\nDescribing the size of the output grid:{text}\nDescribing how to transform the grid:{text}\n'

    return system_prompt


llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",  
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=""
)
json_llm = llm.bind(response_format={"type": "json_object"})

class Config:
    class LanguageGeneration:
        add_meta_hint = True
        example_tasks = 0

    language_generation = LanguageGeneration()

cfg = Config()

class HumanDataGetter():
    def __init__(self):
        self.df = pd.read_csv(CLEAN_DATA_CSV)

    def get_by_task_id(self, task_name):
        candidates = self.df[self.df['task_name'] == task_name]
        candidates = candidates[candidates['solved'] == True]
        return candidates['last_written_solution'].unique().tolist()

def get_answer_by_gpt(task, test_input):
    system_prompt = get_system_prompt(cfg)
    ans = json_llm.invoke([
        (
            "system",
            system_prompt,
        ),
        ("human", f"""Help me solve this puzzle. Here are training examples:
{task['train']}

Here are possible ideas of this puzzle solutions: 
{task['explanations'][:5]}

Now give me the output for this input. Output strictly a json of the grid {{'output': [...]}}: 
{test_input}""")
    ])
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

async def fine_tune_model_with_human_data():
    """Fine-tunes the GPT model using explanations from clean_data.csv."""
    training_data = []
    filenames = os.listdir(TRAINING_DATA_PATH)
    for filename in filenames:
        async with aiofiles.open(os.path.join(TRAINING_DATA_PATH, filename), 'r') as f:
            try:
                task_json = json.loads(await f.read())
                task_name = filename.replace('.json', '')
                task = {
                    'task_name': task_name,
                    'train': task_json['train'],
                }
                task['explanations'] = hd.get_by_task_id(task['task_name'])
                for example in task['train']:
                    system_prompt = get_system_prompt(cfg)
                    prompt = f"""Help me solve this puzzle. Here are training examples:
{task['train']}

Here are possible ideas of this puzzle solutions: 
{task['explanations'][:5]}

Now give me the output for this input. Output strictly a json of the grid {{'output': [...]}}: 
{example['input']}"""
                    completion = json.dumps({'output': example['output']})
                    training_data.append({
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": completion}
                        ]
                    })
            except Exception as e:
                print(f"Error reading training data from {filename}: {e}")
                continue

    fine_tune_file = "fine_tuning_data.jsonl"
    if os.path.exists(fine_tune_file):
        async with aiofiles.open(fine_tune_file, 'r') as f:
            content = await f.read()
            if content.strip():  # If the file is not empty, skip writing
                print(f"{fine_tune_file} already contains data. Skipping write.")
            else:
                async with aiofiles.open(fine_tune_file, 'w') as f:
                    for data in training_data:
                        await f.write(json.dumps(data) + "\n")
    else:
        async with aiofiles.open(fine_tune_file, 'w') as f:
            for data in training_data:
                await f.write(json.dumps(data) + "\n")

    file_response = openai.File.create(
        file=open(fine_tune_file, "rb"),
        purpose='fine-tune'
    )
    training_file_id = file_response['id']

    response = openai.FineTuningJob.create(
        training_file=training_file_id,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={
            "n_epochs": 10
        }
    )
    fine_tune_job_id = response['id']

    while True:
        status_details = openai.FineTuningJob.retrieve(id=fine_tune_job_id)
        status = status_details['status']
        if status == 'succeeded':
            fine_tuned_model = status_details['fine_tuned_model']
            print(f"Fine-tuning completed. Model: {fine_tuned_model}")
            return fine_tuned_model
        elif status == 'failed':
            print("Fine-tuning failed.")
            # Print error messages
            if 'error' in status_details:
                print(f"Error: {status_details['error']['message']}")
            return None
        else:
            print(f"Fine-tuning status: {status}. Waiting...")
            await asyncio.sleep(30)

async def evaluate_model_with_language_generation(model):
    filenames = os.listdir(EVALUATION_DATA_PATH)
    tasks = [process_file_with_language_generation(filename, model) for filename in filenames]

    results = await asyncio.gather(*tasks)
    correct = sum(1 for file_results in results for result in file_results if result['correct'])
    total = sum(len(file_results) for file_results in results if file_results is not None)

    print(f"Accuracy: {correct / total * 100:.2f}%")
    return results

async def process_file_with_language_generation(filename, model):
    async with aiofiles.open(os.path.join(EVALUATION_DATA_PATH, filename), 'r') as f:
        try:
            task_json = json.loads(await f.read())
            task_name = filename.replace('.json', '')
            task = {
                'task_name': task_name,
                'train': task_json['train'],
                'test_inputs': [test_case['input'] for test_case in task_json['test']],
                'test_outputs': [test_case['output'] for test_case in task_json['test']],
            }
            task['explanations'] = hd.get_by_task_id(task['task_name'])
        except Exception as e:
            print(f"Error reading evaluation data from {filename}: {e}")
            return []

    results = []
    for test_input, expected_output in zip(task['test_inputs'], task['test_outputs']):
        system_prompt = get_system_prompt(cfg)
        prompt = f"""Help me solve this puzzle. Here are training examples:
{task['train']}

Here are possible ideas of this puzzle solutions:
{task['explanations'][:5]}

Now give me the output for this input. Output strictly a json of the grid {{'output': [...]}}:
{test_input}"""
        try:
            response = await safe_chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=None
            )
            if response is None:
                print("Failed to get a response after retries.")
                predicted_output = None
            else:
                assistant_response = response['choices'][0]['message']['content']
                # Parse the assistant's response
                try:
                    predicted_output = eval(assistant_response)['output']
                except Exception:
                    print("Failed to parse assistant's response.")
                    predicted_output = None
        except openai.error.OpenAIError as e:
            print(f"An OpenAI error occurred: {e}")
            predicted_output = None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            predicted_output = None

        results.append({
            "input": test_input,
            "expected_output": expected_output,
            "predicted_output": predicted_output,
            "correct": predicted_output == expected_output
        })
    return results

async def main():
    print("Fine-tuning the model...")
    fine_tuned_model = await fine_tune_model_with_human_data()
    if fine_tuned_model is None:
        print("Fine-tuning was skipped due to errors.")
        return

    print("Evaluating the fine-tuned model...")
    await evaluate_model_with_language_generation(fine_tuned_model)

if __name__ == "__main__":
    asyncio.run(main())
