import numpy as np
import traceback
import random

def eval_code(function_declaration, train_cases, test_inputs, q):
    function_declaration = fix_code(function_declaration)
    local_scope = {}
    try:
        exec(function_declaration, None, local_scope)
    except Exception as e:
        q.put({"declaration_error": traceback.format_exc()})
        return

    # Check if transform_grid is defined
    if 'transform_grid' not in local_scope:
        q.put({"error": "Function 'transform_grid' not found in the provided code."})
        return

    testing_result = []
    fails = 0
    passed = 0
    for case in train_cases:
        o = {
            'input': case['input'],
            'target': case['output'],
        }
        inp = np.array(case['input'])
        tgt = np.array(case['output'])
        result = None
        try:
            k = random.random()
            print(f"running code {k}")
            result = local_scope['transform_grid'](inp)
            print(f"finished code {k}")
            o['result'] = result.tolist()
        except Exception as e:
            o['error'] = traceback.format_exc()
        if result is not None and result.shape == tgt.shape and (result == tgt).all():
            passed += 1
            print(f"PASSED CASE! X{passed}/{len(train_cases)}")
        else:
            testing_result.append(o)
            fails += 1
    if fails == 0:
        try:
            answers = [local_scope['transform_grid'](np.array(case)).tolist() for case in test_inputs]
            q.put({'test_answers': answers})
        except Exception as e:
            q.put({"error": traceback.format_exc()})
    else:
        q.put(testing_result)


def fix_code(code):
    code = code.strip()
    parts = code.split("```")
    if len(parts) < 2:
        # If code block wasn't properly formatted, return as is
        return code
    code_block = parts[1].strip()
    if code_block.startswith("python"):
        code_block = code_block[6:].strip()
    return code_block
