import numpy as np
import traceback

def eval_code(function_declaration, train_cases, test_inputs):
    function_declaration = fix_code(function_declaration)
    # print(f"GOT CODE:\n{function_declaration}")
    local_scope = {}
    try:
        exec(function_declaration, None, local_scope)
    except Exception as e:
        return {"declaration_error": traceback.format_exc()}
    # print("VALID FUNCTION!")

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
            result = local_scope['transform_grid'](inp)
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
        return {'test_answers': [ local_scope['transform_grid'](np.array(case)).tolist() for case in test_inputs ]}
    return testing_result


def fix_code(code):
    code = code.strip()
    if code[:9] == "```python":
        code = code[9:]
    if code[-3:] == "```":
        code = code[:-3]
    return code