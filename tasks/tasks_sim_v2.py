from datasets import load_dataset
from typing import List
import yaml

# Load the tasks from the yaml file----------------
yaml_tasks_path = '/mnt/netapp1/Proxecto_NOS/adestramentos/avaliacion/tasks/tasks_ubication.yaml'
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

    def __init__(self, name, dataloader, lang, splitPrompt, cache="./cache"):
        """
        Initializes a Task object.

        Args:
            name (str): The name of the task.
            dataloader (str): The path to the dataloader or to HF ubication.
            lang (str): The language of the task.
            splitPrompt (str): The split prompt for the task.
            cache (str, optional): The cache directory. Defaults to "./cache".
        """
        self.name = name
        self.dataloader = dataloader
        self.lang = lang
        self.dataset = None
        self.splitPrompt = splitPrompt
        self.cache = cache
    
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
        return f"Data(name={self.name}, dataloader={self.dataloader}, lang={self.lang}, splitPrompt={self.splitPrompt}, cache={self.cache})"

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

    def get_correct_option(self, example) -> str:
        """
        Gets the correct option for the given example.

        Args:
            example (dict): The example data.

        Returns:
            str: The correct option.
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
            "dataloaders/belebele.py",
            "gl",
            "RESPOST")
    
    def __init__(self, cache):

        super().__init__(
            "belebele",
            "dataloaders/belebele.py",
            "gl",
            "RESPOST",
            cache)

    def __init__(self, lang, cache):
        super().__init__(
            "belebele",
            None,
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
    

class PAWS(Task):
    """
    Class for the PAWS task.
    """

    def __init__(self):

        super().__init__(
            "paws",
            "dataloaders/paws-gl.py",
            "gl",
            "PUNTUACIO")
    
    def __init__(self, cache):

        super().__init__(
            "paws",
            "dataloaders/paws-gl.py",
            "gl",
            "PUNTUACIO",
            cache)

    def get_correct_option(self, example):

        return str(example["label"])

    def get_options(self, example):

        return ["0","1","2"]

    def build_prompt(self,example, show_answer, show_options = True):

        prompt = ""
        if show_answer:
            prompt = f"""ORACION: {example['sentence1']} 
            PARAFRASE: {example['sentence2']} 
            PUNTUACION: {example['label']}\n"""
        else:
            prompt = f"""ORACION: {example['sentence1']} 
            PARAFRASE: {example['sentence2']} 
            PUNTUACIO"""

        return prompt
    
    def load_data(self):

        self.dataset = load_dataset(self.dataloader,  "paws-gl", cache_dir = self.cache)["test"]
        return self.dataset
    

class OpenBookQA(Task):
    """
    Class for the OpenBookQA task.
    """

    def __init__(self):

        super().__init__(
            "openbookqa",
            "dataloaders/openbookqa.py",
            "gl",
            "RESPOST")
    
    def __init__(self, cache):

        super().__init__(
            "openbookqa",
            "dataloaders/openbookqa.py",
            "gl",
            "RESPOST",
            cache)
    
    def __init__(self, lang, cache):
        super().__init__(
            "openbookqa",
            None,
            lang,
            "RESPOST",
            cache)

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
        self.dataset = load_dataset(hf_dataset, cache_dir = self.cache)["test"]
        print("DATASET CARGADO!")
        return self.dataset
    
class ParafrasesGL(Task):
    """
    Class for the Parafrases-GL task.
    """

    def __init__(self):

        super().__init__(
            "parafrases-gl",
            "dataloaders/parafrases-gl.py",
            "gl",
            "PUNTUACIO")
    
    def __init__(self, cache):

        super().__init__(
            "parafrases-gl",
            "dataloaders/parafrases-gl.py",
            "gl",
            "PUNTUACIO",
            cache)

    def get_correct_option(self, example):

        return str(example["label"])

    def get_options(self, example):

        return ["0","1","2"]

    def build_prompt(self,example, show_answer, show_options = True):

        prompt = ""
        if show_answer:
            prompt = f"""ORACION: {example['sentence1']} 
            PARAFRASE: {example['sentence2']} 
            PUNTUACION: {example['label']}\n"""
        else:
            prompt = f"""ORACION: {example['sentence1']} 
            PARAFRASE: {example['sentence2']} 
            PUNTUACIO"""

        return prompt
    
    def load_data(self):

        self.dataset = load_dataset(self.dataloader,  self.name, cache_dir = self.cache)["test"]
        return self.dataset

class GalCoLA(Task):
    """
    Class for the GalCoLA task.
    """

    def __init__(self):

        super().__init__(
            "galcola",
            "dataloaders/galcola.py",
            "gl",
            "ACEPTABL")
    
    def __init__(self, cache):

        super().__init__(
            "galcola",
            "dataloaders/galcola.py",
            "gl",
            "ACEPTABL",
            cache)

    def get_correct_option(self, example):

        return str(example["is_acceptable"])

    def get_options(self, example):

        return ["0","1"]

    def build_prompt(self,example, show_answer, show_options = True):

        prompt = ""
        if show_answer:
            prompt = f"""ORACION: {example['sentence']} 
            ACEPTABLE: {example['is_aceptable']}\n"""
        else:
            prompt = f"""ORACION: {example['sentence']} 
            ACEPTABL"""

        return prompt
    
    def load_data(self):

        self.dataset = load_dataset(self.dataloader,  "galcola", cache_dir = self.cache)["test"]
        return self.dataset

class SummarizationGL(Task):
    """
    Class for the OpenBookQA task.
    """

    def __init__(self):

        super().__init__(
            "summarization-gl",
            "dataloader/summarization-gl.py",
            "gl",
            "RESUM")
    
    def __init__(self, cache):

        super().__init__(
            "summarization-gl",
            "dataloaders/summarization-gl.py",
            "gl",
            "RESUM",
            cache)
    
    def get_correct_option(self, example):

        return example["summary"]

    def get_options(self, example):

        return [example["summary"]]

    def build_prompt(self,example, show_answer, show_options = True):

        prompt = ""
        if show_options:
            if show_answer:
                prompt = f"""TEXTO: {example['text']}
                RESUMO: {example['summary']}\n"""
            else:
                prompt = f"""TEXTO: {example['text']}
                RESUM"""
        else:
            if show_answer:
                prompt = f"""TEXTO: {example['text']}
                RESUMO: {example['summary']}\n"""
            else:
                prompt = f"""TEXTO: {example['text']}
                RESUM"""
        return prompt
    
    def load_data(self):

        self.dataset = load_dataset(self.dataloader, "summarization-gl", cache_dir = self.cache, split='test[:1000]') #Cargamos só os primeiros 1000 examplos
        return self.dataset