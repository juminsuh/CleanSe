import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import functools
import json
import os
import datasets
import pandas as pd
from datasets import Dataset
from models import load_model

DATA_FOLDER = "/mnt/aix7101/minsuh-dataset"
OPEN_FOLDER = "/home/aix7101/minsuh/intrascore/data/datasets"

def _save_dataset():
    save_path = f'{DATA_FOLDER}/SQuAD'
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
            for sample_id, sample in enumerate(paragraphs): # sample == a single contex with several questions 
                story = sample['context'] # context given
                questions = sample['qas'] # dummies of several questions derived from a single context
                for question_index, question in enumerate(questions): # questions 안의 각각의 question 처리
                    if question["is_impossible"]: # is_impossible == True일 경우 건너뛰기. False일 경우만 아래 코드가 실행된다. 
                        continue
                    dataset['story'].append(story) # append context
                    dataset['question'].append(question['question']) # append a single question
                    dataset['answer'].append({
                        'text': question["answers"][0]['text'], # answers dummies에 있는 answer 중 첫 번째 text answer
                        'answer_start': question["answers"][0]['answer_start'] # 첫 번째 text answer가 시작되는 문자 인덱스
                    })
                    dataset['id'].append(question['id']) # question의 id (매 questions 더미마다 0, 1, 2...으로 초기화될 것)
        dataset_df = pd.DataFrame(dataset)
        print(dataset_df.dtypes)
        dataset = Dataset.from_pandas(dataset_df)

        dataset.save_to_disk(save_path)
    return save_path # 구축한 dataset을 저장한 디렉토리 반환

@functools.lru_cache(1)
def read_all_contexts():
    dataset = datasets.load_from_disk(_save_dataset())
    return {_['id']: _['story'] for _ in dataset}

def get_dataset(tokenizer, split='validation'): # (prompt를 tokenize할 tokenizer, validation dataset을 로드) 
    dataset = datasets.load_from_disk(_save_dataset())
    def encode_coqa(example): # 각 example의 prompt를 구성하고 prompt를 토크나이징하는 함수
        example['answer'] = example['answer']['text'] 
        example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:'
        return tokenizer(prompt, truncation=False, padding=False) # tokenize prompt
    dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False) # dataset의 각 example에 대해 encode_coqa 함수를 적용함, 각 예제를 개별적으로 처리 (batched = False)
    # # dataset에 input_ids와 attention_mask 정보가 추가된다. 
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True) # input_ids, attention_mask 열만 pytorch 텐서 형식으로 포맷된다. 
    return dataset 

def generate_config(tokenizer):

    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889] # _을 ID로 encoding한 후, encoding의 마지막 부분만 (-1)사용
    eos_token_id += [tokenizer.eos_token_id] # [encoded(.), encoded(\n), 29889, tokenizer.eos_token_id(=2)]
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    # Follows Kuhn et al 2023 as Llama does not have CoQA
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in question_framing_ids] # 문자열 eos_token을 토크나이징 -> input_ids를 가져오고 -> 두 번째 토큰을 선택(1)
    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)

if __name__ == '__main__':
    # SAVE_PATH = "/mnt/aix7101/minsuh-dataset/SQuAD-tokenized"
    # MODEL_NAME = "llama-7b-hf"
    # dataset = get_dataset(load_model.load_pretrained_tokenizer(MODEL_NAME))
    # dataset.save_to_disk(SAVE_PATH)
    
    save_path = _save_dataset()
    print(f"successfully saved SQuAD dataset in {save_path}")