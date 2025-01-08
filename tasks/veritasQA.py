from tasks.sim_tasks import Task

class OpenBookQA(Task):
    """
    Class for the OpenBookQA task.
    """
    
    def __init__(self, lang, cache):
        super().__init__(
            "veritasqa",
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