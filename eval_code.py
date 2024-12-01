import numpy as np
import traceback

def eval_code(function_declaration, train_cases, test_inputs):
    function_declaration = """
    
    import numpy as np

def transform_grid(input_grid: np.ndarray) -> np.ndarray:
    # Transforms the input grid by doubling its size and applying specific rules:
    # 1. For each non-zero element in the input grid, create a 2x2 block of the same value
    #    in the corresponding position in the output grid.
    # 2. For each 2x2 block created, draw diagonal lines of '1's (representing blue) 
    #    from the bottom-right corner outward to the bottom-right of the output grid 
    #    and from the top-left corner outward to the top-left of the output grid.

    # Args:
    # input_grid (np.ndarray): A 2D numpy array of integers representing the input grid.

    # Returns:
    # np.ndarray: A 2D numpy array representing the transformed grid.
    # Double the size of the grid
    rows, cols = input_grid.shape
    output_grid = np.zeros((2 * rows, 2 * cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            value = input_grid[i, j]
            if value != 0:
                # Create a 2x2 block of the same value
                output_grid[2*i:2*i+2, 2*j:2*j+2] = value
                
                # Draw diagonal lines from the block bottom-right corner
                for k in range(2*i+2, 2*rows):
                    for l in range(2*j+2, 2*cols):
                        if k-l == (2*i+2)-(2*j+2):
                            output_grid[k, l] = 1

                # Draw diagonal lines from the block top-left corner
                for k in range(0, 2*i):
                    for l in range(0, 2*j):
                        if k-l == 2*i-2*j:
                            output_grid[k, l] = 1

    return output_grid
    
    """
    function_declaration = fix_code(function_declaration)
    print(f"GOT CODE:\n{function_declaration}")
    local_scope = {}
    try:
        exec(function_declaration, None, local_scope)
    except Exception as e:
        return {"declaration_error": traceback.format_exc()}
    print("VALID FUNCTION!")

    testing_result = []
    fails = 0
    for case in train_cases:
        o = {
            'input': case['input'], 
            'target': case['output'],
        }
        inp = np.array(case['input'])
        tgt = np.array(case['output'])
        result = None
        try:
            result = local_scope['transform_grid'](inp)
            o['result'] = result.tolist()
        except Exception as e:
            o['error'] = traceback.format_exc()
        if result is not None and result.shape == tgt.shape and (result == tgt).all():
            print("PASSED CASE!\n"*10)
        else:
            testing_result.append(o)
            fails += 1
    if fails == 0:
        return {'test_answers': [ local_scope['transform_grid'](np.array(case)).tolist() for case in test_inputs ]}
    return testing_result


def fix_code(code):
    code = code.strip()
    if code[:9] == "```python":
        code = code[9:]
    if code[-3:] == "```":
        code = code[:-3]
    return code