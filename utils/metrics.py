import torch
from scipy.spatial.distance import cosine
import evaluate
import numpy as np
import logging
import os

def cosine_score(tokenizer, model, sentence1, sentence2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer.pad_token = tokenizer.eos_token
    inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True).to(device)
    inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True).to(device)

    # Embedding generation
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # Use only the last hidden state
    embeddings1 = outputs1.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    embeddings2 = outputs2.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    # Cosine similarity
    similarity_score = 1 - cosine(embeddings1, embeddings2)
    return similarity_score

def mover_score(bert_model, generation, reference):
    os.environ['MOVERSCORE_MODEL'] = bert_model
    from utils.moverscore_v2 import sentence_score

    moverscore = sentence_score(generation, [reference], trace=False)
    return moverscore

def bert_score(language, bert_model, generations, references, print_results=False):
    logging.info(f"Evaluating BERT Score...")
    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(predictions=generations, references=references, 
                                model_type= bert_model, idf=True, num_layers = 11, lang=language)
    if print_results:
        for i in range(len(generations)):
            print(f'Reference {i}: {references[i]}')
            print(f'Generated {i}: {generations[i]}')
            print(f'Bert Score {i}: [precision: {np.mean(bertscore_results["precision"][i]).round(4)}, recall: {np.mean(bertscore_results["recall"][i]).round(4)}, f1: {np.mean(bertscore_results["f1"][i]).round(4)}]')
            print(f'-----------------------')
    return bertscore_results


