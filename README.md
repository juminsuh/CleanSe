## 🧼 Cleanse

Accepted to ACL 2025 GEM workshop (poster & oral)

![Image](https://github.com/user-attachments/assets/d8d6e3b2-ee89-4119-b2f9-fc9859cf1c46)

### Abstract

Despite the outstanding performance of large language models (LLMs) across various NLP tasks, hallucinations in LLMs–where LLMs generate inaccurate responses–remains as a critical problem as it can be directly connected to a crisis of building safe and reliable LLMs. Uncertainty estimation is primarily used to measure hallucination level in LLM responses so that correct and incorrect answers can be distinguished clearly. This study proposes an effective uncertainty estimation approach, Clustering-based semantic consistency (Cleanse). Cleanse quantifies the uncertainty with the proportion of the intra-cluster consistency in the total consistency between LLM hidden embeddings which contain adequate semantic information of generations, by employing clustering. The effectiveness of Cleanse for detecting hallucination is validated using four off-the-shelf models, LLaMA-7B, LLaMA-13B, LLaMA2-7B and Mistral-7B and two question-answering benchmarks, SQuAD and CoQA.

### How to implement 

```bash
pip install -r requirements.txt
```

```bash
sh run.sh
```
