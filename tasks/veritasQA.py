from tasks_sim_v2 import Task
from datasets import load_dataset
import yaml
import os
import random

yaml_tasks_path = f'{os.path.dirname(__file__)}/tasks_ubication.yaml'
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None

# Function to convert list of dictionaries to a single dictionary
def convert_to_dict(list_of_dicts):
    return {k: v for d in list_of_dicts for k, v in d.items()}

class VeritasQA(Task):
    """
    Class for the VeritasQA task.
    """
    
    def __init__(self, lang, cache):
        super().__init__(
            "veritasqa",
            None,
            lang,
            "RESPOST",
            cache)

    def get_correct_option(self, example):
        raise NotImplementedError("This method is not implemented for this task")
    
    def get_correct_options(self, example):
        return example["correct_answers"].split(";")

    def get_incorrect_options(self, example):
        return example["incorrect_answers"].split(";")
    
    def get_options(self, example):
        correct_options = example["correct_answers"].split(";")
        incorrect_options = example["incorrect_answers"].split(";")
        options = correct_options + incorrect_options
        options = random.shuffle(correct_options)
        return options
    
    def get_best_answer(self, example):
        return example['best_answer']
    
    def build_prompt(self,example, show_answer, show_options):

        prompt = ""
        options = self.get_options(example)
        if show_options:
            if show_answer:
                correct_option = self.get_best_answer(example)
                prompt = f"""CONTEXTO: {example['question']} 
                PREGUNTA: {example['question']} 
                {"\n".join([f"OPCION {i+1}: {option}" for i, option in enumerate(options)])}
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""CONTEXTO: {example['question']} 
                PREGUNTA: {example['question']} 
                {"\n".join([f"OPCION {i+1}: {option}" for i, option in enumerate(options)])}
                RESPOST"""
        else:
            if show_answer:
                correct_option = self.get_best_answer(example)
                prompt = f"""CONTEXTO: {example['question']} 
                PREGUNTA: {example['question']}
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""CONTEXTO: {example['question']} 
                PREGUNTA: {example['question']}
                RESPOST"""
        return prompt
    
    def load_data(self):
        data_yaml = load_yaml(yaml_tasks_path)
        openbookqa_dict = convert_to_dict(data_yaml['veritasqa'])
        hf_dataset = openbookqa_dict[self.lang]
        print(f"DATASET: {hf_dataset}")
        if hf_dataset=="None":
            exit(f"Dataset not found for {self.name} and language {self.lang}")
        
        repo=hf_dataset[0]
        lang_subset=hf_dataset[1]
        print(f"DATASET: {repo} - {lang_subset}")
        self.dataset = load_dataset(repo, lang_subset, cache_dir = self.cache)["test"]

        print("DATASET CARGADO!")
        return self.dataset