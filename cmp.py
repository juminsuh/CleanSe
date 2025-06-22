import pickle
import difflib
import pprint

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compare_pickles(pkl1, pkl2, max_depth=2):
    def _stringify(obj, depth=0):
        if depth > max_depth:
            return str(type(obj))
        if isinstance(obj, dict):
            return {k: _stringify(v, depth + 1) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ _stringify(v, depth + 1) for v in obj ]
        elif isinstance(obj, tuple):
            return tuple(_stringify(v, depth + 1) for v in obj)
        else:
            return obj

    obj1 = load_pickle(pkl1)
    obj2 = load_pickle(pkl2)

    str1 = pprint.pformat(_stringify(obj1), indent=2, width=120)
    str2 = pprint.pformat(_stringify(obj2), indent=2, width=120)

    print("\n📌 Difference between the two .pkl files:\n")
    for line in difflib.unified_diff(
        str1.splitlines(), str2.splitlines(), fromfile=pkl1, tofile=pkl2, lineterm=""
    ):
        print(line)

# 사용 예시
file1 = "/mnt/aix7101/minsuh-output/llama-7b-hf_SQuAD_for_clustering/0.pkl"
file2 = "/mnt/aix7101/minsuh-output/llama-7b-hf_SQuAD_for_clustering_2/0.pkl"
compare_pickles(file1, file2)
