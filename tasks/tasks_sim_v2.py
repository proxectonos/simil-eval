from datasets import DownloadConfig, load_dataset
from typing import Union, List
import yaml
import os

# Load the tasks from the yaml file----------------
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

#--------------------------------------------------

class Task:
    """
    Base class for different tasks.
    """

    def __init__(self, name, lang, splitPrompt, cache="./cache", token=""):
        """
        Initializes a Task object.

        Args:
            name (str): The name of the task.
            lang (str): The language of the task.
            splitPrompt (str): The split prompt for the task.
            cache (str, optional): The cache directory. Defaults to "./cache".
            token (str, optional): HF token to access private datasets. Defaults to "".
        """
        self.name = name
        self.lang = lang
        self.dataset = None
        self.splitPrompt = splitPrompt
        self.cache = cache
        self.token = token
    
    def set_cache(self, cache):
        """
        Sets the cache directory.

        Args:
            cache (str): The cache directory.
        """
        self.cache = cache

    def set_splitPrompt(self, splitPrompt):
        """
        Sets the split prompt.

        Args:
            splitPrompt (str): The split prompt.
        """
        self.splitPrompt = splitPrompt

    def __str__(self):
        """
        Returns a string representation of the Task object.

        Returns:
            str: The string representation of the Task object.
        """
        return f"Data(name={self.name}, lang={self.lang}, splitPrompt={self.splitPrompt}, cache={self.cache})"

    def build_prompt(self, example, show_answer, show_options) -> str:
        """
        Builds the prompt for the given example.

        Args:
            example (dict): The example data.
            show_answer (bool): Whether to include the answer in the prompt.
            show_options (bool): Whether to include the options in the prompt.

        Returns:
            str: The built prompt.
        """
        raise NotImplementedError 
    
    def load(self):
        """
        Loads the dataset for the task.

        Returns:
            dataset: The loaded dataset.
        """
        raise NotImplementedError

    def get_correct_option(self, example) -> Union[str, List[str]]:
        """
        Gets the correct option for the given example.

        Args:
            example (dict): The example data.

        Returns:
            str|List[str]: The correct or the correct options.
        """
        raise NotImplementedError
    
    def get_options(self, example) -> List:
        """
        Gets the options for the given example.

        Args:
            example (dict): The example data.

        Returns:
            list: The options.
        """
        raise NotImplementedError
    

class Belebele(Task):
    """
    Class for the Belebele task.
    """

    def __init__(self):

        super().__init__(
            "belebele",
            "gl",
            "RESPOST")
    
    def __init__(self, cache):

        super().__init__(
            "belebele",
            "gl",
            "RESPOST",
            cache)

    def __init__(self, lang, cache):
        super().__init__(
            "belebele",
            lang,
            "RESPOST",
            cache)
        
    def get_correct_option(self, example):

        answers_dict = {"1":"mc_answer1",
                "2":"mc_answer2",
                "3":"mc_answer3",
                "4":"mc_answer4"}

        return example[answers_dict[example["correct_answer_num"]]]

    def get_options(self, example):

        return [example["mc_answer1"], example["mc_answer2"], example["mc_answer3"], example["mc_answer4"]]

    def build_prompt(self,example, show_answer, show_options):

        prompt = ""
        options = self.get_options(example)
        if show_options:
            if show_answer:
                correct_option = self.get_correct_option(example)
                prompt = f"""CONTEXTO: {example['flores_passage']} 
                PREGUNTA: {example['question']} 
                OPCION 1: {options[0]}
                OPCION 2: {options[1]}
                OPCION 3: {options[2]}
                OPCION 4: {options[3]}
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""CONTEXTO: {example['flores_passage']} 
                PREGUNTA: {example['question']} 
                OPCION 1: {options[0]}
                OPCION 2: {options[1]}
                OPCION 3: {options[2]}
                OPCION 4: {options[3]}
                RESPOST"""
        else:
            if show_answer:
                correct_option = self.get_correct_option(example)
                prompt = f"""CONTEXTO: {example['flores_passage']} 
                PREGUNTA: {example['question']}
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""CONTEXTO: {example['flores_passage']} 
                PREGUNTA: {example['question']}
                RESPOST"""
        return prompt

    
    def load_data(self):
        data_yaml = load_yaml(yaml_tasks_path)
        openbookqa_dict = convert_to_dict(data_yaml['belebele'])
        hf_dataset = openbookqa_dict[self.lang]
        print(f"DATASET: {hf_dataset}")
        if hf_dataset=="None":
            exit(f"Dataset not found for {self.name} and language {self.lang}")
        elif isinstance(hf_dataset, list): #Belebele subsets from original Belebele HF repo
            repo=hf_dataset[0]
            subset=hf_dataset[1]
            print(f"DATASET: {repo} - {subset}")
            self.dataset = load_dataset(repo, subset, cache_dir = self.cache)["test"]
        else: #Galician belebele
            self.dataset = load_dataset(hf_dataset, cache_dir = self.cache)["train"]
        print("DATASET CARGADO!")
        return self.dataset
    
    

class OpenBookQA(Task):
    """
    Class for the OpenBookQA task.
    """

    def __init__(self):

        super().__init__(
            "openbookqa",
            "gl",
            "RESPOST")
    
    def __init__(self, cache):

        super().__init__(
            "openbookqa",
            "gl",
            "RESPOST",
            cache)
    
    def __init__(self, lang, cache, token):
        super().__init__(
            "openbookqa",
            lang,
            "RESPOST",
            cache,
            token)

    def get_correct_option(self, example):

        answers_dict = {"A":0,
                "B":1,
                "C":2,
                "D":3}

        return example["choices"]["text"][answers_dict[example["answerKey"]]]

    def get_options(self, example):

        return [example["choices"]["text"][0], example["choices"]["text"][1], example["choices"]["text"][2], example["choices"]["text"][3]]

    def build_prompt(self,example, show_answer, show_options = True):

        prompt = ""
        options = self.get_options(example)
        if show_options:
            if show_answer:
                correct_option = self.get_correct_option(example)
                prompt = f"""PREGUNTA: {example['question_stem']} 
                OPCION A: {options[0]}
                OPCION B: {options[1]}
                OPCION C: {options[2]}
                OPCION D: {options[3]}
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""PREGUNTA: {example['question_stem']} 
                OPCION A: {options[0]}
                OPCION B: {options[1]}
                OPCION C: {options[2]}
                OPCION D: {options[3]}
                RESPOST"""
        else:
            if show_answer:
                correct_option = self.get_correct_option(example)
                prompt = f"""PREGUNTA: {example['question_stem']} 
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""PREGUNTA: {example['question_stem']} 
                RESPOST"""
        return prompt
    
    def load_data(self):
        data_yaml = load_yaml(yaml_tasks_path)
        openbookqa_dict = convert_to_dict(data_yaml['openbookqa'])
        hf_dataset = openbookqa_dict[self.lang]
        print(f"DATASET: {hf_dataset}")
        if hf_dataset=="None":
            exit(f"Dataset not found for {self.name} and language {self.lang}")
        self.dataset = load_dataset(hf_dataset, cache_dir = self.cache, download_config=DownloadConfig(token=self.token))["test"]
        print("DATASET CARGADO!")
        return self.dataset
    
class VeritasQA(Task):
    """
    Class for the VeritasQA task.
    """
    
    def __init__(self, lang, cache):
        super().__init__(
            "veritasqa",
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
            formated_options = "\n".join([f"OPCION {i+1}: {option}" for i, option in enumerate(options)])
            if show_answer:
                correct_option = self.get_best_answer(example)
                prompt = f"""CONTEXTO: {example['question']}
                PREGUNTA: {example['question']} 
                {formated_options}
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""CONTEXTO: {example['question']} 
                PREGUNTA: {example['question']} 
                {formated_options}
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
        self.dataset = load_dataset(repo, cache_dir = self.cache)[lang_subset]
        return self.dataset
