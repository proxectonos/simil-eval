from typing import List
import yaml

# Load the tasks from the yaml file----------------
yaml_tasks_path = f'./configs/tasks_ubication.yaml'
def load_yaml():
    with open(yaml_tasks_path, 'r') as file:
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

class SimilarityTask:
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
    


    






    
