from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import evaluate
import tasks.tasks_sim_v2 as tasks_sim_v2
import torch
from scipy.spatial.distance import cosine
import random
import csv
import numpy as np
import argparse
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
#-- Cálculo da similaridade ----------------------------------------------

#- Cálculo da similaridade entre dúas oracións ----------------------------------------------
def cosine_score(tokenizer, model, sentence1, sentence2):
    # Tokenização
    tokenizer.pad_token = tokenizer.eos_token
    inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)

    # gerando embeddings para cada frase
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # usamos só a última camada
    embeddings1 = outputs1.last_hidden_state.mean(dim=1).squeeze().numpy()
    embeddings2 = outputs2.last_hidden_state.mean(dim=1).squeeze().numpy()

    #  coseno
    similarity_score = 1 - cosine(embeddings1, embeddings2)
    return similarity_score

def mover_score(task,generation, reference):
    os.environ['MOVERSCORE_MODEL'] = bertmodels_yaml[task.lang]
    from moverscore_v2 import sentence_score

    moverscore = sentence_score(generation, [reference], trace=False)
    return moverscore

def compute_sentence_similarity(task,metric, tokenizer, model, sentence1, sentence2):

    if metric == "cosine":
        return cosine_score(tokenizer, model, sentence1, sentence2)
    elif metric == "moverscore":
        return mover_score(task,sentence1, sentence2)
    else:
        raise NotImplementedError

#- Cálculo da similaridade entre dous corpus de oracións ----------------------------------------------
def bert_score(task,generations, references, print_results=False):
    logging.info(f"Evaluating BERT Score...")
    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(predictions=generations, references=references, 
                                model_type= bertmodels_yaml[task.lang], idf=True, num_layers = 11, lang="gl")
    if print_results:
        for i in range(len(generations)):
            print(f'Reference {i}: {references[i]}')
            print(f'Generated {i}: {generations[i]}')
            print(f'Bert Score {i}: [precision: {np.mean(bertscore_results["precision"][i]).round(4)}, recall: {np.mean(bertscore_results["recall"][i]).round(4)}, f1: {np.mean(bertscore_results["f1"][i]).round(4)}]')
            print(f'-----------------------')
    final_information = f'[precision: {np.mean(bertscore_results["precision"]).round(4)}, recall: {np.mean(bertscore_results["recall"]).round(4)}, f1: {np.mean(bertscore_results["f1"]).round(4)}, hashcode: {bertscore_results["hashcode"]}]'
    logging.info(final_information)
    print(f'-----------------------')
    return final_information
    
def compute_corpus_similarity(task, metric, generations, references):
    if metric == "bertscore":
        return bert_score(task, generations, references)
    else:
        raise NotImplementedError

def get_generated_answer(task,answer):
    generated_answer =""
    lines = answer[0].split('\n')
    for line in reversed(lines):
        if re.match(fr'^\s*{task.splitPrompt}.*', line):
            generated_answer = line.strip()
            generated_answer = re.sub(fr'^{task.splitPrompt}.*:', '', generated_answer)
            break
    return generated_answer

def evaluate_sentence_similarity(task, metric, tokenizer, model, dataset, csv_reader, fewshots_examples_ids):
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
            similarity = compute_sentence_similarity(task,metric, tokenizer, model, original_answer, generated_answer) if generated_answer else 0.0 #Check if generated answer is empty (stange but it can happen)
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

def evaluate_corpus_similarity(task, metric, dataset, csv_reader, fewshots_examples_ids):
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

def evaluate_similarity(task, metrics, model_id, results_file, tokenHF):
    
    logging.info("\nStarting similarity evaluation between generated and original answers...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=task.cache, use_auth_token=tokenHF)
    model = AutoModel.from_pretrained(model_id, cache_dir=task.cache, use_auth_token=tokenHF)
    dataset = task.load_data()
    global_metrics_information = ""
    for metric in metrics:
        with open(results_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            fewshots_examples_ids = next(csv_reader)
            fewshots_examples_ids = [int(id) for id in fewshots_examples_ids]
            
            logging.info(f"Evaluating metric {metric}...")
            if metric in ["cosine","moverscore"]:
                metric_information = evaluate_sentence_similarity(task, metric, tokenizer, model, dataset, csv_reader, fewshots_examples_ids)
                global_metrics_information += metric_information
            elif metric in ["bertscore"]:
                metric_information = evaluate_corpus_similarity(task, metric, dataset, csv_reader, fewshots_examples_ids)
                global_metrics_information += metric_information
            else:
                raise NotImplementedError
    separator = "-----------------------------------------------------------\n"
    logging.info(f"\n{separator}----METRICS SUMMARY----\n{global_metrics_information}\n{separator}")
            

#- Xeración de respostas ----------------------------------------------
def generate_answers_no_pad(model, tokenizer, prompt_ids):
    max_new_tokens = 20 if task.name != "summarization-gl" else 100
    final_outputs = model.generate(prompt_ids, 
        do_sample=True,
        max_new_tokens=max_new_tokens,
        repetition_penalty=0.5 if task.name != "summarization-gl" else 1.2, 
        temperature=0.5)
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True) 

def generate_answers(model, tokenizer, prompt_ids):
    max_new_tokens = 20 if task.name != "summarization-gl" else 100
    final_outputs = model.generate(prompt_ids, 
        do_sample=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.config.eos_token_id,
        repetition_penalty=0.5 if task.name != "summarization-gl" else 1.2, 
        temperature=0.5)
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)  

def xerar_e_gardar_textos(task, model_id, results_file_name, examples_file, cache, tokenHF):      
    answers = []
    fewshots_examples_ids = []
    logging.info(f'Generating texts for model {model_id}...')
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, use_auth_token=tokenHF)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache, use_auth_token=tokenHF)
    if model_id == "irlab-udc/Llama-3.1-8B-Instruct-Galician":
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
        
        for i, prompt in enumerate(csv_reader):
            if i in fewshots_examples_ids:
                continue
            prompt = prompt[0]
            prompt_ids = tokenizer.encode(f'{prompt}', return_tensors='pt')
            
            if len(prompt_ids[0])+100 > max_position_embeddings:
                fewshots_examples_ids.append(i) #Trick to avoid missing examples during similarity evaluation
                print(f"ID: {i} - Pass due to length of the prompt")
            else:
                generated_sequence = generation_function(model, tokenizer, prompt_ids)
                parts = generated_sequence.rsplit(fr'{task.splitPrompt}.*:', 1)
                answer = parts[-1].strip()
                answers.append(generated_sequence)
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

#- Construcción das preguntas----------------------------------------------
def generate_examples(task, examples_file, fewshot_num=5, show_options=True):
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

#- Test de novas funcionalidades ----------------------------------------------
def test():
    print("Test function")
    return

#-- Execución ----------------------------------------------

# Custom action to handle --token without a value
class OptionalString(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, values)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Evaluation of QA datasets using similarity')
    # General arguments
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate (Belebele only)')
    parser.add_argument('--model', type=str, help='Model to use for text generation')
    parser.add_argument('--cache', type=str, help='Directory where cache data will be stored')
    parser.add_argument('--token', type=str, nargs='?', default=None, action=OptionalString, help='Hugging Face authentication token')
    parser.add_argument('--test', action='store_true', help='Test functionalities')
    parser.add_argument('--language', type=str, help='Dataset language')
    # Example creation arguments
    parser.add_argument('--create_examples', action='store_true', help='Generate examples')
    parser.add_argument('--fewshot_num', type=int, help='Number of few-shots to generate (default is 5)')
    parser.add_argument('--examples_file', type=str, help='Name of the examples file')
    parser.add_argument('--show_options', type=lambda x: (str(x).lower() == 'true'), help='Include answer options when generating examples')
    # Generation arguments
    parser.add_argument('--generate_answers', action='store_true', help='Generate answers for the created examples')
    parser.add_argument('--results_file', type=str, help='Name of the results file')
    # Evaluation arguments
    parser.add_argument('--evaluate_similarity', action='store_true', help='Evaluate similarity between generated and original answers')
    parser.add_argument('--metrics', type=str, nargs='+', help='Metrics to use for similarity evaluation')

    args = parser.parse_args()
    print(args)

    if args.test:
        print("Test funcionalities")
        test()
        exit()

    if args.dataset == "belebele":
        task = tasks_sim_v2.Belebele(lang=args.language, cache=args.cache)
    
    elif args.dataset == "paws":
        task = tasks_sim_v2.PAWS(cache=args.cache)

    elif args.dataset == "openbookqa":
        task = tasks_sim_v2.OpenBookQA(lang=args.language, cache=args.cache)

    elif args.dataset == "paraphrasis":
        task = tasks_sim_v2.ParafrasesGL(cache=args.cache)

    elif args.dataset == "cola":
        task = tasks_sim_v2.GalCoLA(cache=args.cache)

    elif args.dataset == "summarization":
        task = tasks_sim_v2.SummarizationGL(cache=args.cache)

    else:
        exit("Task not supported. Currently implemented tasks are [PAWS, Belebele, OpenBookQA, ParafrasesGL, GalCoLA, Summarization-GL]")

    if args.create_examples:
        generate_examples(task, examples_file=args.examples_file, fewshot_num=args.fewshot_num, show_options=args.show_options)
    
    if args.generate_answers:
        xerar_e_gardar_textos(task, args.model, args.results_file, args.examples_file, args.cache, args.token)
    
    if args.evaluate_similarity:
        supported_metrics = ["cosine","bertscore","moverscore"]
        for metric in args.metrics:
            if metric not in supported_metrics:
                exit(f"Unsupported metric: {metric}. Currently implemented metrics are {supported_metrics}")
        evaluate_similarity(task, args.metrics, args.model, args.results_file, args.token)
        exit()