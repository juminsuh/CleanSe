import functools
import json
import os
import sys

import datasets
import pandas as pd
from datasets import Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import settings as settings

OPEN_FOLER = "/home/aix7101/minsuh/cleanse/data/datasets"

def save_dataset():
    save_path = f'{settings.DATA_FOLDER}/coqa_dataset' 
    if not os.path.exists(save_path):
        with open(f'{OPEN_FOLER}/coqa-dev-v1.0.json', 'r') as infile: 
            data = json.load(infile)['data']

        dataset = {}

        dataset['story'] = []
        dataset['question'] = []
        dataset['answer'] = []
        dataset['id'] = []

        for _, sample in enumerate(data):
            story = sample['story']
            questions = sample['questions']
            answers = sample['answers']
            for question_index, question in enumerate(questions):
                dataset['story'].append(story)
                dataset['question'].append(question['input_text'])
                dataset['answer'].append({
                    'text': answers[question_index]['input_text'],
                    'answer_start': answers[question_index]['span_start']
                })
                dataset['id'].append(sample['id'] + '_' + str(question_index))

        dataset_df = pd.DataFrame.from_dict(dataset)

        dataset = Dataset.from_pandas(dataset_df)

        dataset.save_to_disk(save_path)
    return save_path

@functools.lru_cache(1)
def read_all_contexts():
    dataset = datasets.load_from_disk(save_dataset())
    return {_['id']: _['story'] for _ in dataset}

def get_dataset(tokenizer, split='validation'):
    dataset = datasets.load_from_disk(save_dataset())
    
    def encode_coqa(example):
        example['answer'] = example['answer']['text']
        example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:' # prompt
        return tokenizer(prompt, truncation=False, padding=False)

    dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def generate_config(tokenizer):
    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889] 
    eos_token_id += [tokenizer.eos_token_id]
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in question_framing_ids]
    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)

if __name__ == '__main__':
    save_path = save_dataset()
    print(f"successfully saved coqa dataset in {save_path}.")