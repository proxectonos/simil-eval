from core.SimilarityTask import SimilarityTask
from core.SimilarityTask import load_yaml, convert_to_dict
from datasets import load_dataset


class Xstorycloze(SimilarityTask):
    """
    Class for the XStoryCloze task.
    """
    
    def __init__(self, lang, cache):
        super().__init__(
            "xstorycloze",
            lang,
            "CONTINUAC",
            cache)

        # Define the field mappings for different languages (gl, pt have strange fields names)
        self.field_mappings = {
            "gl": {
                "story_id": "InputStoryid",
                "answer_right_ending": "AnswerRightEnding",
                "input_sentence_1": "InputSentence1",
                "input_sentence_2": "InputSentence2",
                "input_sentence_3": "InputSentence3",
                "input_sentence_4": "InputSentence4",
                "sentence_quiz1": "RandomFifthSentenceQuiz1",
                "sentence_quiz2": "RandomFifthSentenceQuiz2"
            },
            "pt": {
                "answer_right_ending": "AnswerRightEnding",
                "input_sentence_1": "InputSentence1",
                "input_sentence_2": "InputSentence2",
                "input_sentence_3": "InputSentence3",
                "input_sentence_4": "InputSentence4",
                "sentence_quiz1": "RandomFifthSentenceQuiz1",
                "sentence_quiz2": "RandomFifthSentenceQuiz2"
            },
            "default": {
                "answer_right_ending": "answer_right_ending",
                "input_sentence_1": "input_sentence_1",
                "input_sentence_2": "input_sentence_2",
                "input_sentence_3": "input_sentence_3",
                "input_sentence_4": "input_sentence_4",
                "sentence_quiz1": "sentence_quiz1",
                "sentence_quiz2": "sentence_quiz2"
            }
        }

    def get_field(self, field_name):
    # Get the appropriate field name based on the language
        return self.field_mappings.get(self.lang, self.field_mappings["default"]).get(field_name)

    def get_correct_option(self, example):

        right_ending = example[self.get_field("answer_right_ending")]
        if right_ending == 1:
            return example[self.get_field("sentence_quiz1")]
        elif right_ending == 2:
            return example[self.get_field("sentence_quiz2")]
        else:
            raise ValueError(f'Right ending not found. Check the example {example["story_id"]} in the dataset.')

    def get_options(self, example):
        return [
            example[self.get_field("sentence_quiz1")],
            example[self.get_field("sentence_quiz2")]
        ]
    
    def build_input_text(self, example):
        return f"{example[self.get_field('input_sentence_1')]} {example[self.get_field('input_sentence_2')]} {example[self.get_field('input_sentence_3')]} {example[self.get_field('input_sentence_4')]}"

    def build_prompt(self,example, show_answer, show_options = True):
        prompt = ""
        input_text = self.build_input_text(example)
        options = self.get_options(example)
        if show_options:
            if show_answer:

                correct_option = self.get_correct_option(example)
                prompt = f"""TEXTO: {input_text} 
                OPCION 1: {options[0]}
                OPCION 2: {options[1]}
                CONTINUACION: {correct_option}\n"""
            else:
                prompt = f"""TEXTO: {input_text}
                OPCION 1: {options[0]}
                OPCION 2: {options[1]}
                CONTINUAC"""
        else:
            if show_answer:
                correct_option = self.get_correct_option(example)
                prompt = f"""TEXTO: {input_text}  
                CONTINUACION: {correct_option}\n"""
            else:
                prompt = f"""TEXTO: {input_text} 
                CONTINUAC"""
        return prompt
    
    def load_data(self): #Using only first 500 examples to evaluate a similar number of examples for all tasks
        data_yaml = load_yaml()
        data_dict = convert_to_dict(data_yaml['xstorycloze'])
        hf_dataset = data_dict[self.lang]
        print(f"Dataset: {hf_dataset}")
        if hf_dataset=="None":
            exit(f"Dataset not found for {self.name} and language {self.lang}")
        elif isinstance(hf_dataset, list): #en, es are in the same repository
            repo=hf_dataset[0]
            subset=hf_dataset[1]
            print(f"Dataset: {repo} - {subset}")
            self.dataset = load_dataset(repo, subset, cache_dir = self.cache)["eval"].select(range(500)) 
        else:
            if self.lang == "cat":
                self.dataset = load_dataset(hf_dataset, cache_dir = self.cache)["eval"].select(range(500)) 
            else: #gl, pt  name "test" instead of "eval"
                self.dataset = load_dataset(hf_dataset, cache_dir = self.cache)["test"].select(range(500)) 
        return self.dataset