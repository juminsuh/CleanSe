import numpy as np
import pickle as pkl
from rouge_score import rouge_scorer
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from metric import *

rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def printInfo(resultDict):
    print(len(resultDict))
    for item in resultDict:
        for key in item.keys():
            print(key)
        exit()

def get_threshold(thresholds, tpr, fpr):
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    return thresholdOpt

def getAccuracy(Label, Score, thresh):
    count = 0
    for ind, item in enumerate(Score):
        if item>=thresh and Label[ind]==1:
            count+=1
        if item<thresh and Label[ind]==0:
            count+=1
    return count/len(Score)

def getAcc(resultDict, file_name):
    correctCount = 0
    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        if rougeScore>0.9:
            correctCount += 1
    print("Acc:", 1.0*correctCount/len(resultDict))

def getPCC(x, y):
    rho = np.corrcoef(np.array(x), np.array(y))
    return rho[0,1]

def getCluster(resultDict, resultDict_v2):
    whether_correct = []
    for item in resultDict:
        ansGT = item['answer']
        generations = item['most_likely_generation']
        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        if rougeScore > 0.9:
            whether_correct.append(1)
        else:
            whether_correct.append(0)
    correct_cluster = 0
    correct_count = 0
    wrong_cluster = 0
    wrong_count = 0
    for idx, item in enumerate(resultDict_v2):
        number_of_clusters = item['num_of_cluster']
        if whether_correct[idx] == 1: 
            correct_cluster += number_of_clusters
            correct_count += 1
        else: 
            wrong_cluster += number_of_clusters
            wrong_count += 1
    avg_correct_cluster = correct_cluster/correct_count
    avg_wrong_cluster = wrong_cluster/wrong_count
    # avg of num of cluster when generation is correct, avg of num of cluster when generation is wrong
    print(f"avg_correct_cluster: {avg_correct_cluster}, avg_wrong_cluster: {avg_wrong_cluster}")

def getAUROC(resultDict, resultDict_v2, file_name):
    Label = []
    Score = []
    Perplexity = []
    LexicalSimilarity = []
    Entropy = []
    CosineScore = []
    CleanseScore = []

    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        Perplexity.append(-item["perplexity"])
        Entropy.append(-item["entropy"])
        LexicalSimilarity.append(item["lexical_similarity"])
        CosineScore.append(item['cosine_score'])

        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        if rougeScore>0.9:
            Label.append(1) # non-hallucinate (correct)
        else:
            Label.append(0) # hallucinate (incorrect)
        Score.append(rougeScore)
    
    for item in resultDict_v2:
        CleanseScore.append(item['cleanse_score'])    

######### AUROC ###########
    fpr, tpr, thresholds = roc_curve(Label, Perplexity)
    AUROC = auc(fpr, tpr)
    thresh_Perplexity = get_threshold(thresholds, tpr, fpr)
    print("AUROC-Perplexity:", AUROC)

    fpr, tpr, thresholds = roc_curve(Label, Entropy)
    AUROC = auc(fpr, tpr)
    thresh_Entropy = get_threshold(thresholds, tpr, fpr)
    print("AUROC-Entropy:", AUROC)

    fpr, tpr, thresholds = roc_curve(Label, LexicalSimilarity)
    AUROC = auc(fpr, tpr)
    thresh_LexicalSim = get_threshold(thresholds, tpr, fpr)
    print("AUROC-LexicalSim:", AUROC)

    fpr, tpr, thresholds = roc_curve(Label, CosineScore)
    AUROC = auc(fpr, tpr)
    thresh_CosineScore = get_threshold(thresholds, tpr, fpr)
    print("AUROC-CosineScore:", AUROC)
    
    fpr, tpr, thresholds = roc_curve(Label, CleanseScore)
    AUROC = auc(fpr, tpr)
    thresh_CleanseScore = get_threshold(thresholds, tpr, fpr)
    print("threshold-CleanseScore:", thresh_CleanseScore)
    print("AUROC-CleanseScore:", AUROC)
    
    rho_Perplexity = getPCC(Score, Perplexity)
    rho_Entropy = getPCC(Score, Entropy)
    rho_LexicalSimilarity = getPCC(Score, LexicalSimilarity)
    rho_CosineScore = getPCC(Score, CosineScore)
    rho_CleanseScore = getPCC(Score, CleanseScore)
    print("rho_Perplexity:", rho_Perplexity)
    print("rho_Entropy:", rho_Entropy)
    print("rho_LexicalSimilarity:", rho_LexicalSimilarity)
    print("rho_CosineScore:", rho_CosineScore)
    print("rho_CleanseScore:", rho_CleanseScore)
    
    acc = getAccuracy(Label, Perplexity, thresh_Perplexity)
    print("Perplexity Accuracy:", acc)
    acc = getAccuracy(Label, Entropy, thresh_Entropy)
    print("Entropy Accuracy:", acc)
    acc = getAccuracy(Label, LexicalSimilarity, thresh_LexicalSim)
    print("LexicalSimilarity Accuracy:", acc)
    acc = getAccuracy(Label, CosineScore, thresh_CosineScore)
    print("CosineScore Accuracy:", acc)
    acc = getAccuracy(Label, CleanseScore, thresh_CleanseScore)
    print("CleanseScore Accuracy:", acc)

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

if __name__ == "__main__":

    file_name = "/mnt/aix7101/minsuh-output/llama-7b-hf_SQuAD_for_clustering/0.pkl"
    # file_name = "/mnt/aix7101/minsuh-output/llama-13b-hf_SQuAD_for_clustering/0.pkl"
    # file_name = "/mnt/aix7101/minsuh-output/llama-7b-hf_coqa_for_clustering_final/0.pkl"
    # file_name = "/mnt/aix7101/minsuh-output/llama-13b-hf_coqa_for_clustering/0.pkl"
    # file_name = "/mnt/aix7101/minsuh-output/Llama-2-7b-hf_SQuAD_for_clustering_1/0.pkl"    
    # file_name = "/mnt/aix7101/minsuh-output/llama2-7b-hf_coqa_for_clustering/0.pkl"
    # file_name = "/mnt/aix7101/minsuh-output/mistral-7b_SQuAD_for_clustering/0.pkl"
    # file_name = "/mnt/aix7101/minsuh-output/mistral-7b_coqa_for_clustering/0.pkl"

    f = open(file_name, "rb")
    resultDict = pkl.load(f)

    file_name_v2 = f"/mnt/aix7101/minsuh-output/llama-7b-hf_SQuAD_nli-deberta-v3-base/0.pkl"
    f_v2 = open(file_name_v2, "rb")
    resultDict_v2 = pkl.load(f_v2)

    # printInfo(resultDict)
    print(f"model: {file_name.split('/')[4]}, dataset: {file_name.split('_')[1]}")
    getAcc(resultDict, file_name)
    getAUROC(resultDict, resultDict_v2, file_name)
    getCluster(resultDict, resultDict_v2)
    

