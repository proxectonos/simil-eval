from abc import ABC, abstractmethod

class SurprisalTask(ABC):
    """
    Base class for surprisal tasks.

    Attributes:
        lang (str): The language of the dataset.
        cache_dir (str): The directory where the dataset cache is stored.
    """
    def __init__(self, lang, cache_dir):
        """
        Initializes the SurprisalTask with the specified language and cache directory.

        Args:
            lang (str): The language of the dataset.
            cache_dir (str): The directory where the dataset cache is stored.
        """
        self.lang = lang
        self.cache_dir = cache_dir
    
    @abstractmethod
    def load_evaluation_dataset(self):
        """
        Loads the evaluation dataset. This method change a lot depending on the task.
        """
        pass
    