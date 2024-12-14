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
#Hint: You may want to use the following guidance to implement the function:
#{hypothesis}   
def code_gen_prompt(train_examples, hypothesis):
    return [
    ("system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
    ('user', f"""{format_train_examples(train_examples)}
Now, please write a python program transform_grid(input_grid: np.ndarray[int]) -> np.ndarray[int] that
transforms the input grid to the corresponding output grid.
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
    # If there's a declaration error or a top-level error, handle it and return immediately
    if isinstance(results, dict):
        if 'declaration_error' in results:
            return (f"Unfortunately this code didn't work. The code declaration failed.\n"
                    f"Here is the error:\n{results['declaration_error']}\n"
                    "Please fix the declaration and try again.\n"
                    "Just reply with the implementation of transform_grid(...) in Python and nothing else.")
        if 'error' in results:
            return (f"Unfortunately this code didn't work. Here is the error:\n{results['error']}\n"
                    "Please fix the error and try again.\n"
                    "Just reply with the implementation of transform_grid(...) in Python and nothing else.")

        # If 'test_answers' is here, that means it succeeded, so no formatting needed.
        if 'test_answers' in results:
            return "The code provided a test answer, but we expected some failures to format."

        # If we get here, check if results is something else unexpected
        # Possibly it's a dictionary with failing test cases?
        # If it's a dictionary with keys not recognized, just return a general message.
        # But normally failing test cases are a list of dicts, not a single dict.
        if not isinstance(results, list):
            # Unrecognized structure
            return ("Unfortunately, the code didn't provide a valid set of results.\n"
                    "Just reply with the implementation of transform_grid(...) in Python and nothing else.")

    # If results is a list, we assume it's a list of test case results (failures)
    if isinstance(results, list):
        res = "Unfortunately this code didn't work. Here are the errors it made:\n"
        for i, example in enumerate(results):
            res += f"Case {i}:\n"
            if 'input' in example:
                res += f"Input:\n{plt_mx(example['input'])}\n"
            if 'target' in example:
                res += f"Target Output:\n{plt_mx(example['target'])}\n"
            if 'result' in example:
                res += f"Code Output:\n{plt_mx(example['result'])}\n"
            if 'error' in example:
                res += f"Error from running the code: {example['error']}\n"
        res += ("Please try again and correct these errors.\n"
                "Just reply with the implementation of transform_grid(...) in Python and nothing else.")
        return res

    # If none of the above conditions matched, handle gracefully:
    return ("Unfortunately, the code didn't provide a recognizable result format.\n"
            "Just reply with the implementation of transform_grid(...) in Python and nothing else.")
