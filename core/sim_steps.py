from core.sim_tasks import Task
import utils.metrics as sim_metrics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np
import random
import csv
import re
import os
import logging
import yaml

# Config settings file  ----------------
yaml_bertmodels_path = f'./configs/bert_models.yaml'
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None
bertmodels_yaml = load_yaml(yaml_bertmodels_path)

class EvaluatingModel():
    def __init__(self, model_id, cache, tokenHF):
        self.model_id = model_id
        self.cache = cache
        self.tokenHF = tokenHF
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

#-- Computation of similarity metrics ----------------------------------------------
def compute_sentence_similarity(task:Task, metric, tokenizer, model, sentence1, sentence2):

    if metric == "cosine":
        return sim_metrics.cosine_score(tokenizer, model, sentence1, sentence2)
    elif metric == "moverscore":
        bert_model = bertmodels_yaml[task.lang]
        return sim_metrics.mover_score(bert_model,sentence1, sentence2)
    else:
        raise NotImplementedError
   
def compute_corpus_similarity(task:Task, metric, generations, references):
    if metric == "bertscore":
        bert_model = bertmodels_yaml[task.lang]
        bertscore_results = sim_metrics.bert_score(task.lang, bert_model, generations, references, print_results=False)
        final_information = f'[precision: {np.mean(bertscore_results["precision"]).round(4)}, recall: {np.mean(bertscore_results["recall"]).round(4)}, f1: {np.mean(bertscore_results["f1"]).round(4)}, hashcode: {bertscore_results["hashcode"]}]'
        logging.info(final_information)
        print(f'-----------------------')
        return final_information
    else:
        raise NotImplementedError

def get_generated_answer(task:Task, answer):
    generated_answer =""
    lines = answer[0].split('\n')
    for line in reversed(lines):
        if re.match(fr'^\s*{task.splitPrompt}.*', line):
            generated_answer = line.strip()
            generated_answer = re.sub(fr'^{task.splitPrompt}.*:', '', generated_answer)
            break
    return generated_answer

def evaluate_sentence_similarity(task:Task, model, tokenizer, metric,dataset, csv_reader, fewshots_examples_ids):
    similarities = []
    correct_similarities = []
    offset_fewshot = 0
    correct_answers = 0
    for i, answer in enumerate(csv_reader):
        if i in fewshots_examples_ids: offset_fewshot += 1
        example = dataset[i+offset_fewshot] # Get the i-th example of the dataset with the correction of fewshot examples
        generated_answer = get_generated_answer(task, answer)
        answer_similarities = []
        correct_option = task.get_correct_option(example)
        original_options = task.get_options(example)
        print(f"ID  {i} - Generated answer: {generated_answer}")
        j=1
        for original_answer in original_options:
            similarity = compute_sentence_similarity(task, metric, tokenizer, model, original_answer, generated_answer) if generated_answer else 0.0 #Check if generated answer is empty (stange but it can happen)
            answer_similarities.append(similarity)
            if original_answer == correct_option:
                correct_similarities.append(similarity)
            print(f"    Similarity score with option {j}: {original_answer}: {similarity}")
            j +=1
        if max(answer_similarities) == correct_similarities[-1] and max(answer_similarities) > 0.0: #Remove 0.0 similarity with all cases:
            correct_answers += 1
        similarities.append(np.mean(answer_similarities))
        print(f"    Mean score with question {i}: {similarities[-1]}")
        print(f"    Score with correct option '{correct_option}': {correct_similarities[-1]}")
    final_information= f"--{metric.upper()} RESULTS--\n"
    final_information+= f"Global Mean similarity score: {np.mean(similarities)}\n"
    final_information+= f"Global Mean similarity score with correct options: {np.mean(correct_similarities)}\n"
    final_information+= f"Percentage of correct answers (over 1): {correct_answers/len(similarities)}\n"
    logging.info(final_information)
    print(f"---------------------------------")
    return final_information

def evaluate_corpus_similarity(task:Task, metric, dataset, csv_reader, fewshots_examples_ids):
    offset_fewshot = 0
    generated_answer = []
    correct_options = []
    original_options = []
    for i, answer in enumerate(csv_reader):
        if i in fewshots_examples_ids: offset_fewshot += 1
        example = dataset[i+offset_fewshot] # Get the i-th example of the dataset with the correction of fewshot examples
        generated_answer.append(get_generated_answer(task, answer))
        correct_options.append(task.get_correct_option(example))
        original_options.append(task.get_options(example))

    logging.info(f"--{metric.upper()} RESULTS----------------")
    final_information = f"--{metric.upper()} RESULTS--\n"
    final_information+= f"Similarity with correct options: "
    metric_values = compute_corpus_similarity(task, metric, generated_answer, correct_options)
    final_information+= metric_values+"\n"
    final_information+= f"Similarity with all options: "
    metric_values = compute_corpus_similarity(task, metric, generated_answer, original_options)
    final_information+= metric_values
    logging.info(final_information)
    print(f"---------------------------------")
    return final_information

def evaluate_similarity(task:Task, evaluated_model:EvaluatingModel, metrics, results_file):
    
    logging.info("\nStarting similarity evaluation between generated and original answers...")
    dataset = task.load_data()
    model = AutoModel.from_pretrained(evaluated_model.model_id, cache_dir=evaluated_model.cache, use_auth_token=evaluated_model.tokenHF)
    tokenizer = AutoTokenizer.from_pretrained(evaluated_model.model_id, cache_dir=evaluated_model.cache, use_auth_token=evaluated_model.tokenHF)
    model.to(evaluated_model.device)
    global_metrics_information = ""
    for metric in metrics:
        with open(results_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            fewshots_examples_ids = next(csv_reader)
            fewshots_examples_ids = [int(id) for id in fewshots_examples_ids]
            
            logging.info(f"Evaluating metric {metric}...")
            if metric in ["cosine","moverscore"]:
                metric_information = evaluate_sentence_similarity(task, model, tokenizer, metric, dataset, csv_reader, fewshots_examples_ids)
                global_metrics_information += metric_information
            elif metric in ["bertscore"]:
                metric_information = evaluate_corpus_similarity(task, metric, dataset, csv_reader, fewshots_examples_ids)
                global_metrics_information += metric_information
            else:
                raise NotImplementedError
    separator = "-----------------------------------------------------------\n"
    logging.info(f"\n{separator}----METRICS SUMMARY----\n{global_metrics_information}\n{separator}")
            

#- Xeración de respostas ----------------------------------------------
def generate_answers_no_pad(model, task:Task, tokenizer, prompt_ids):
    max_new_tokens = 20 if task.name != "None" else 100
    final_outputs = model.generate(**prompt_ids, 
        do_sample=True,
        max_new_tokens=max_new_tokens,
        repetition_penalty=0.5 if task.name != "None" else 1.2, 
        temperature=0.5)
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True) 

def generate_answers(model, task:Task, tokenizer, prompt_ids):
    max_new_tokens = 20 if task.name != "None" else 100
    final_outputs = model.generate(**prompt_ids, 
        do_sample=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.config.eos_token_id,
        repetition_penalty=0.5 if task.name != "None" else 1.2, 
        temperature=0.5)
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)  

def generate_completions(task:Task, evaluated_model:EvaluatingModel, results_file_name, examples_file):      
    answers = []
    fewshots_examples_ids = []
    logging.info(f'Generating texts for model {evaluated_model.model_id}...')
    model = AutoModelForCausalLM.from_pretrained(evaluated_model.model_id, cache_dir=evaluated_model.cache, use_auth_token=evaluated_model.tokenHF)
    tokenizer = AutoTokenizer.from_pretrained(evaluated_model.model_id, cache_dir=evaluated_model.cache, use_auth_token=evaluated_model.tokenHF)
    model.to(evaluated_model.device)

    if evaluated_model.model_id in ["irlab-udc/Llama-3.1-8B-Instruct-Galician","meta-llama/Llama-3.1-8B-Instruct"]:
        generation_function = generate_answers_no_pad
    else:
        generation_function = generate_answers

    with open(examples_file,"r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        fewshots_examples_ids = next(csv_reader)
        # Check if the config has max_position_embeddings attribute
        if hasattr(model.config, 'max_position_embeddings'):
            max_position_embeddings = model.config.max_position_embeddings
        else:
            max_position_embeddings = 2048
        print("Maximum block size: ", max_position_embeddings)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for i, prompt in enumerate(csv_reader):
            if i in fewshots_examples_ids:
                continue
            prompt = prompt[0]
            prompt_ids = tokenizer(f'{prompt}', return_tensors='pt').to(device)
            
            if len(prompt_ids[0])+100 > max_position_embeddings:
                fewshots_examples_ids.append(i) #Trick to avoid missing examples during similarity evaluation
                print(f"ID: {i} - Pass due to length of the prompt")
            else:
                generated_sequence = generation_function(model, task, tokenizer, prompt_ids)
                answers.append(generated_sequence)
                parts = generated_sequence.rsplit(fr'{task.splitPrompt}.*:', 1)
                answer = parts[-1].strip()
                print(f"ID: {i} - {answer}")

    with open(results_file_name, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(fewshots_examples_ids)
        for i, answer in enumerate(answers):
            if i in fewshots_examples_ids:
                continue
            csv_writer.writerow([answer])    
    
    logging.info(f'Generation finished! Texts saved in {results_file_name}...')
    print("----------------------------------------------------------------------")

#-----------------------------------------------
def create_examples(task, examples_file, fewshot_num=5, show_options=True):
    if os.path.exists(examples_file):
        logging.warning(f"The file {examples_file} already exists. Skipping example creation...")
        return
    
    if not fewshot_num: fewshot_num = 0 # Correction to no fewshot configurations
    task.load_data()
    fewshots_examples_ids =  random.sample(list(range(0,len(task.dataset))),fewshot_num)
    logging.info(f"Fewshots examples: {fewshots_examples_ids}")
    
    fewshot_examples = ""
    for k in fewshots_examples_ids:
        example = task.dataset[k]
        fewshot_examples += task.build_prompt(example, show_answer=True, show_options=show_options)
    with open(examples_file,"w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(fewshots_examples_ids)
        for i, data in enumerate(task.dataset):
            if i in fewshots_examples_ids:
                continue
            question = fewshot_examples + task.build_prompt(data, show_answer=False, show_options=show_options)
            csv_writer.writerow([question])
    logging.info(f"Exemples created succesfully! Examples saved in {examples_file}")

def test():
    print("Test function")
    return
