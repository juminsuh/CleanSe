import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import functools
import json
import os
import datasets
import pandas as pd
from datasets import Dataset
from models import load_model
import settings
from settings import *

OPEN_FOLDER = "/home/aix7101/minsuh/cleanse/data/datasets"

def _save_dataset():
    save_path = f'{settings.DATA_FOLDER}/SQuAD'
    if not os.path.exists(save_path):
        with open('{}/dev-v2.0.json'.format(OPEN_FOLDER), 'r') as infile:
            data = json.load(infile)['data']

        dataset = {}

        dataset['story'] = []
        dataset['question'] = []
        dataset['answer'] = []
        dataset['id'] = []

        for _data in data:
            paragraphs = _data["paragraphs"]
            for sample_id, sample in enumerate(paragraphs): 
                story = sample['context'] 
                questions = sample['qas'] 
                for question_index, question in enumerate(questions):
                    if question["is_impossible"]: 
                        continue
                    dataset['story'].append(story) 
                    dataset['question'].append(question['question']) 
                    dataset['answer'].append({
                        'text': question["answers"][0]['text'],
                        'answer_start': question["answers"][0]['answer_start'] 
                    })
                    dataset['id'].append(question['id'])
        dataset_df = pd.DataFrame(dataset)
        print(dataset_df.dtypes)
        dataset = Dataset.from_pandas(dataset_df)

        dataset.save_to_disk(save_path)
    return save_path

@functools.lru_cache(1)
def read_all_contexts():
    dataset = datasets.load_from_disk(_save_dataset())
    return {_['id']: _['story'] for _ in dataset}

def get_dataset(tokenizer, split='validation'):  
    dataset = datasets.load_from_disk(_save_dataset())
    def encode_squad(example): 
        example['answer'] = example['answer']['text'] 
        example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:' # prompt
        return tokenizer(prompt, truncation=False, padding=False) 
    dataset = dataset.map(encode_squad, batched=False, load_from_cache_file=False) 
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True) 
    return dataset 

def generate_config(tokenizer):

    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889] 
    eos_token_id += [tokenizer.eos_token_id]
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in question_framing_ids] 
    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)

if __name__ == '__main__':
    
    save_path = _save_dataset()
    print(f"successfully saved SQuAD dataset in {save_path}")