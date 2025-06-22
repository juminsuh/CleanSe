# import streamlit as st 
from models import load_model
from func.metric import *
import dataeval.SQuAD as SQuAD
import dataeval.coqa as coqa
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

# @st.cache_resource
def get_tokenizer(model_name):
    return load_model.load_pretrained_tokenizer(model_name)

# @st.cache_resource
def get_model(model_name):
    return load_model.load_pretrained_model(model_name)

# @st.cache_resource
def get_clustering_model(device):
    model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    return tokenizer, model

def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens

def cleanse(input_text, generated_texts, last_embeddings, total, tokenizer, model, device):

    if last_embeddings is None: # None cannot be subscriptable
        cleanse_score = 0.0  
    else: 
        clusters = [[{'answer': generated_texts[0], 'embeddings': last_embeddings[0, :]}]]

        # clustering with nli model
        for i in range(1, len(generated_texts)):

            already_add = False
            for c in clusters:
                qa_1 = input_text + ' ' + generated_texts[i] # answer which should be included to a certain cluster
                qa_2 = input_text + ' ' + c[0]['answer'] # first member of cluster c
                
                inputs = tokenizer(qa_1, qa_2, return_tensors="pt", padding=True, truncation=True)
                inputs = {key:value.to(device) for key, value in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                
                predicted_label = torch.argmax(logits, dim=1)
                
                reverse_inputs = tokenizer(qa_2, qa_1, return_tensors="pt", padding=True, truncation=True)
                reverse_inputs = {key:value.to(device) for key, value in reverse_inputs.items()}
                
                with torch.no_grad():
                    reverse_outputs = model(**reverse_inputs)
                    reverse_logits = reverse_outputs.logits
                
                reverse_predicted_label = torch.argmax(reverse_logits, dim=1)

                if predicted_label.item() == 1 and reverse_predicted_label.item() == 1: # if bi-directional entailment then add 
                    c.append({'answer': generated_texts[i], 'embeddings': last_embeddings[i, :]})
                    already_add = True
                    break
            
            if not already_add:
                clusters.append([{'answer': generated_texts[i], 'embeddings': last_embeddings[i]}]) # create new cluster
                
        # compute cleanse score
        total_similarity = total
        intra_similarity = 0.0
        for c in clusters:
            num_cluster = len(c)
            if num_cluster < 2:
                continue 
            for i in range(num_cluster):
                for j in range(i+1, num_cluster): # cosine similarity between outputs within the same cluster
                    intra_similarity += cosine_similarity_v2(c[i]['embeddings'], c[j]['embeddings'])
        cleanse_score = intra_similarity / total_similarity
        
        return cleanse_score
                
def main(dataset, model_name, input_text):

    device = "cuda:0"
    
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)

    # if model_name == "llama-13b-hf":
    #     model.to(device)
    #     dispatch_model(model, device_map = config.device_map)
    
    cluster_tokenizer, cluster_model = get_clustering_model(device)
    
    input = tokenizer(input_text, return_tensors="pt")
    print(f"input: {input}")
    input_ids = input['input_ids'].to(device)
    print(f"input_ids: {input_ids}")
    attention_mask = input['attention_mask'].to(device)
    print(f"attention_mask: {attention_mask}")
    input_length = input_ids.shape[1]
    print(f"input_length: {input_length}")
    # generation_config = get_generation_config(input_ids, tokenizer, dataset) 
    # generation_config = transformers.GenerationConfig(**generation_config)    

    dict_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                num_beams=1,
                                do_sample=False,
                                # generation_config=generation_config,
                                max_new_tokens=256,
                                pad_token_id=tokenizer.eos_token_id,
                                output_hidden_states = True,
                                return_dict_in_generate=True,
                                output_scores=True) # returns logits for each token generation which consists generated sequence

    most_likely_generation = dict_outputs.sequences.cpu()[0, input_length:] 
    output_text = tokenizer.decode(most_likely_generation, skip_special_tokens=True)
    print(f"output_text: {output_text}")


    dict_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                num_beams=1,
                                do_sample=True,
                                num_return_sequences=10,
                                # generation_config=generation_config,
                                max_new_tokens=256,
                                pad_token_id=tokenizer.eos_token_id,
                                early_stopping=True, 
                                top_p=0.99,
                                top_k=5,
                                temperature=0.5,
                                output_hidden_states = True,
                                return_dict_in_generate=True)

    print(f"Generated sequence shape: {dict_outputs.sequences.shape}")
    generation = dict_outputs.sequences[:, input_length:].cpu()
    print(f"generation: {generation.shape}")
    generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generation]
    num_tokens = get_num_tokens(generation)
    print(f"num_tokens: {num_tokens}")

    hidden_states = dict_outputs.hidden_states # tuple
    # print(f"hidden_states: {hidden_states}")
    print(f"len hidden_states: {len(hidden_states)}")
    last_embeddings = extract_embeddings(hidden_states, num_tokens)
    print(f"last_embeddings: {last_embeddings}")
    print(f"len last_embeddings: {last_embeddings.shape}")

    _, total = compute_CosineSimilarity(last_embeddings)
    score = cleanse(input_text, generated_texts, last_embeddings, total, cluster_tokenizer, cluster_model, device)
    
    return output_text, score

if __name__ == '__main__':
    
    threshold = {'llama-7b-hf-SQuAD': 0, 'llama-7b-hf-coqa': 0.5357,
             'llama-13b-hf-SQuAD': 0.5446, 'llama-13b-hf-coqa': 0.5033,
             'Llama-2-7b-hf-SQuAD': 0.5369, 'Llama-2-7b-hf-coqa': 0.4858,
             'mistral-7b-SQuAD': 0.8585, 'mistral-7b-coqa': 0.6473}
    
    dataset = 'SQuAD'
    model_name = "Llama-2-7b-hf"
    input_text = '''Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. 
    Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), 
    which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy". Q: In what R&B group was she the lead singer? A: '''
    response, score = main(dataset, model_name, input_text)
    print("Output (Most Likely Generation)")
    print("-------------------------------")
    print(response)
    print("Cleanse Score")
    print("-------------------------------")
    print(f"{score:.3f}")
    
    key = f"{model_name}-{dataset}"
    if score < threshold[key]:
        print("⚠️ High chance of hallucination!")
    else:
        print("✅ Likely factual.")
    
    
