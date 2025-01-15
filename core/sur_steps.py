from utils.surprisal import get_surprisal, get_surprisal_last_word
from utils.surprisal import get_scorer
import numpy as np

def get_surprisal_scorer(model_name, cache_dir, tokenHF):
    model_scorer = get_scorer(model_name, cache_dir, tokenHF)
    return model_scorer

def surprisal_score_calame(model_scorer, model_name, dataset):
    scores = []
    for examples in dataset:
        score = get_surprisal_last_word(model_scorer, examples)
        scores.append(score)

    scores_mean = round(np.mean([x for x in scores if x != None]),4)
    print(f"""Results for model: {model_name}
Mean score last word: {scores_mean}
{"#"*40}""")

def surprisal_score_cola(model_scorer, model_name, dataset_good, dataset_bad):
    good_scores = []
    bad_scores = []
    for good in dataset_good:
        good_score = get_surprisal(model_scorer, good)
        good_scores.append(good_score)
    
    for bad in dataset_bad:
        bad_score = get_surprisal(model_scorer, bad)
        bad_scores.append(bad_score)

    good_mean = round(np.mean(good_scores),4)
    bad_mean = round(np.mean(bad_scores),4)
    difference_mean = round(bad_mean - good_mean,4)
    print(f"""Results for model: {model_name}
Good mean: {good_mean}
Bad mean: {bad_mean}
Difference between means (bad-good): {difference_mean}
{good_mean},{bad_mean},{difference_mean}
{"#"*40}""")
    
def surprisal_score_xlni(model_scorer, model_name, dataset_contradiction, dataset_entailment, dataset_neutral):
    contradiction_scores = []
    entailment_scores = []
    neutral_scores = []
    for contradiction in dataset_contradiction:
        contradiction_score = get_surprisal(model_scorer, contradiction)
        contradiction_scores.append(contradiction_score)
    
    for entailment in dataset_entailment:
        entailment_score = get_surprisal(model_scorer, entailment)
        entailment_scores.append(entailment_score)
    
    for neutral in dataset_neutral:
        neutral_score = get_surprisal(model_scorer, neutral)
        neutral_scores.append(neutral_score)

    contradiction_mean = round(np.mean(contradiction_scores),4)
    entailment_mean = round(np.mean(entailment_scores),4)
    neutral_mean = round(np.mean(neutral_scores),4)
    print(f"""Results for model: {model_name}
Contradiction mean: {contradiction_mean}
Entailment mean: {entailment_mean}
Neutral mean: {neutral_mean}
{"#"*40}""")
    
def test():
    print("Test function")
    return
