import argparse
import glob
import json
import os
import time
import pandas as pd
import torch
import tqdm
import transformers
from accelerate import dispatch_model
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import settings as settings
import dataeval.SQuAD as SQuAD
import dataeval.coqa as coqa
from models import load_model
import utils
import config
from func.metric import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-7b-hf')
parser.add_argument('--dataset', type=str, default='SQuAD')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=str, default='for_clustering')

args = parser.parse_args()
logInfo = open(f"{settings.GENERATION_FOLDER}/logInfo_{args.model}_{args.dataset}.txt", mode="w",encoding="utf-8")

def get_dataset_fn(data_name):
    if data_name == "SQuAD":
        return SQuAD.get_dataset
    if data_name == "coqa":
        return coqa.get_dataset

def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    if data_name == 'SQuAD':
        generation_config = SQuAD.generate_config(tokenizer)
    if data_name == 'coqa':
        generation_config = coqa.generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config

@torch.no_grad()
def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=args.num_generations_per_prompt):
    
    device = args.device
    
    model = load_model.load_pretrained_model(model_name)
    tokenizer = load_model.load_pretrained_tokenizer(model_name)

    if model_name == "llama-13b-hf":
        model.to(device)
        dispatch_model(model, device_map = config.device_map)
    
    # for name, param in model.named_parameters():
    #     print(f"Layer {name} is on device: {param.device}")

    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}

    sequences = []
    for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch['id'][0] in old_sequences: 
            sequences.append(old_sequences[batch['id'][0]])
            continue
        
        # feed each sample in dataset to the model as input
        input_ids = batch['input_ids'].to(device) 
        attention_mask = batch['attention_mask'].to(device)
        
        input_length = input_ids.shape[1] # dimension of each input_ids
        generation_config = get_generation_config(input_ids, tokenizer, args.dataset) 
        generation_config = transformers.GenerationConfig(**generation_config)
        if args.decoding_method == 'beam_search':
            raise NotImplementedError() 
        elif args.decoding_method == 'greedy':
            dict_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                        num_beams=1,
                                        do_sample=False,
                                        generation_config=generation_config,
                                        output_hidden_states = True,
                                        return_dict_in_generate=True,
                                        output_scores=True) # returns logits for each token generation which consists generated sequence

            scores = dict_outputs.scores    #([logits for token1] ,[logits for token2],[logits for token3]...) | (seq_len, vocab_size) | len(scores): seq_len, len(scores[0]) = vocab_size = 32000
            perplexity = get_perplexity_score(scores) # baseline: perplexity
            most_likely_generations = dict_outputs.sequences.cpu()[0, input_length:] 

        torch.cuda.empty_cache()
        generations = []
        num_gens = args.num_generations_per_prompt
        # generate 10 sequences per sample
        while num_gens > 0:
            dict_outputs =  model.generate(input_ids=input_ids, attention_mask=attention_mask,
                            num_beams=1, num_return_sequences=min(max_num_gen_once, num_gens),
                            do_sample=True, top_p=args.top_p, top_k=args.top_k,
                            temperature=args.temperature, generation_config=generation_config,
                            output_hidden_states = True, return_dict_in_generate=True, output_scores=True
                            )
            # dict_outputs.shape: [num_return_sequences, generation_length]
            scores = dict_outputs.scores
            generation = dict_outputs.sequences[:, input_length:].cpu() # slice so that only token ids of answer part are left (batch_size, seq_len)
            generations.append(generation) 
            num_tokens = get_num_tokens(generation) # the number of generated tokens per generation
            
            predictive_entropy = get_lenghthNormalized_entropy(scores, num_tokens) # baseline: LN entropy

            hidden_states = dict_outputs.hidden_states # tuple
            last_embeddings = extract_embeddings(hidden_states, num_tokens) # middle embedding of last token for each generation
            cosine_score, total = compute_CosineSimilarity(last_embeddings)

            num_gens -= len(generation)

        generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
        generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
        generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
        lexical_similarity = getLexicalSim(generated_texts) # baseline: lexical similarity

        # remember the data
        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['id'][0],
            question=batch['question'][0],
            answer=batch['answer'][0],
            additional_answers=[],
        )
        curr_seq.update(
            dict(
                most_likely_generation_ids = most_likely_generations,
                generations_ids=generations,
            )
        )
        curr_seq.update(
            dict(
                most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                generations=generated_texts,
            )
        )
        curr_seq.update(
            dict(
                last_embeddings = last_embeddings
            )
        )
        curr_seq.update(
            dict(
                perplexity=perplexity
            )
        )
        curr_seq.update(
            dict(
                lexical_similarity=lexical_similarity
            )
        )
        curr_seq.update(
            dict(
                entropy=predictive_entropy
            )
        )
        curr_seq.update(
            dict(
                cosine_score = cosine_score
            )
        )
        curr_seq.update(
            dict(
                cosine_total = total
            )
        )
        sequences.append(curr_seq)
        torch.cuda.empty_cache()

        print("Question:", batch['question'][0])
        print("AnswerGT:", batch['answer'][0])
        print("MostLikelyAns:", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True))
        print("Batch_Generations:", generated_texts)
        print("Perplexity:", perplexity)
        print("NormalizedEntropy: ", predictive_entropy)
        print("LexicalSimilarity: ", lexical_similarity)
        print("CosineScore:", cosine_score)

        # below are recorded to logInfo
        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True), file=logInfo)
        print("Question:", batch['question'][0], file=logInfo)
        print("GTAns:", batch['answer'][0], file=logInfo)
        print("BestAns:", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True), file=logInfo)
        print("BatchGenerations:", generated_texts, file=logInfo)
        print("Perplexity:", perplexity, file=logInfo)
        print("NormalizedEntropy: ", predictive_entropy, file=logInfo)
        print("LexicalSimilarity: ", lexical_similarity, file=logInfo)
        print("CosineScore:", cosine_score, file = logInfo)
        print("\n","\n","\n", file=logInfo)
    return sequences

def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens # type is list(): [# of tokens of genenation_1, # of tokens of genenation_2...# of tokens of genenation_10], len(num_tokens) = 10

def main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from: # start where the process was over before
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.project_ind}') # 최종 output 저장 경로
        os.makedirs(cache_dir, exist_ok=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
        run_id = len(old_results)
        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    sequences = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences)
    print(f'Writing {len(sequences)} generations to {cache_dir}...')
    pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
    return

if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess) 
