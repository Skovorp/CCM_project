def direct_answer_prompt(train_examples, explanations, test_input):
    return [
            ("system", "You are a genius solving puzzles."),
            ("human", f"""Help me solve this puzzle. Here are training examples:
{train_examples}

Here are possible ideas of this puzzle solutions: 
{explanations}

Now give me the output for this input. Output stricly a json of the grid {{'output': [...]}}: 
{test_input}""")
]

def hypothesis_synth_prompt(train_examples, hypotheses):
    return [
        ("system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
    ("user",f"""I will give you training examples and some
    human hypotheses and i want you to synthesize a new one that would
    best describe a natural language solution of the puzzle.
    The number in the input grid can be mapped to the following colors:0:black; 1:blue; 2:red; 3:green;
    4:yellow; 5:grey; 6:fuschia; 7:orange; 8:teal; 9:brown. The goal is to understand the underlying pattern of the input grid
    and use that on the output grid. I want you to crate a human language solution 
    based on the training examples and the human hypotheses that solve them.
    The training examples are:{format_train_examples(train_examples)}. The human hypotheses are: {hypotheses}.
    Please generate new one that will encapture all the information. Please just give me the synthesized hypothesis
    and nothing else.
"""),
]
    
def code_gen_prompt(train_examples, hypothesis):
    return [
    ("system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
    ('user', f"""{format_train_examples(train_examples)}
Now, please write a python program transform_grid(input_grid: np.ndarray[int]) -> np.ndarray[int] that
transforms the input grid to the corresponding output grid.
Hint: You may want to use the following guidance to implement the function:
{hypothesis}
The number in the input grid can be mapped to the following colors:0:black; 1:blue; 2:red; 3:green;
4:yellow; 5:grey; 6:fuschia; 7:orange; 8:teal; 9:brown
Just reply with the implementation of transform_grid(input_grid: np.ndarray[int]) in Python and nothing
else, each cell in the output should only be numbers from 0 to 9. Don't do any hardcoding of the matrices.
Write generalizable code."""),
]

def plt_mx(matrix):
    return "\n".join(["[" + " ".join(map(str, row)) + "]" for row in matrix])
    
def format_train_examples(input_cases):
    results = []
    for i, case in enumerate(input_cases):
        case_str = f"Case {i}:\n"
        case_str += "Input:\n" + plt_mx(case["input"]) + "\n"
        case_str += "Target Output:\n" + plt_mx(case["output"]) + "\n"
        
        results.append(case_str)
    
    return "\n".join(results)


def format_code_run_results(results, hypothesis=None):
    res = "Unfortunately this code didnt work. Here are the errors it made.\n"
    if 'declaration_error' in results:
        res += f"The code declaration didnt work. Here is the error it produced: {results['declaration_error']}"
        return res 
    for i, example in enumerate(results):
        res += f"Case {i}:\n"
        res += f"Input:\n{plt_mx(example['input'])}\n"
        res += f"Target Output:\n{plt_mx(example['target'])}\n"
        if 'result' in example:
            res += f"Code Output:\n{plt_mx(example['result'])}\n"
        elif 'error' in example:
            res += f"Error from running the code: {example['error']}"
    res += f"""Please try again and correct these errors.
Just reply with the implementation of transform_grid(input_grid: np.ndarray[int]) in Python and nothing else, each cell in the output should only be numbers from 0 to 9. """
# Again, you may want to use the following guidance to implement the function: {hypothesis} 
    return res