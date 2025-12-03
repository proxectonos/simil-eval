from core.SurprisalTask import SurprisalTask
from datasets import load_dataset

class Globalpiqa(SurprisalTask):
    def __init__(self, lang, cache_dir):
        self.lang = lang
        self.cache_dir = cache_dir
        self.dataset_good = []
        self.dataset_bad = []
        self.language_subsets = {
            "gl": "glg_latn",
            "en": "eng_latn",
            "cat": "cat_latn",
            "es": "spa_latn_spai",
            "pt": "por_latn_port"
        }

    def load_evaluation_dataset(self):
        language_code = self.language_subsets.get(self.lang)
        dataset = load_dataset('mrlbenchmarks/global-piqa-nonparallel', language_code)['test']
        dataset_good = dataset_bad = []
        for item in dataset:
            if item['label'] == 0:
                dataset_good.append(item['solution0'])
                dataset_bad.append(item['solution1'])
            else:
                dataset_good.append(item['solution1'])
                dataset_bad.append(item['solution0'])
        self.dataset_good = dataset_good
        self.dataset_bad = dataset_bad