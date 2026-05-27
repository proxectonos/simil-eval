# Extending Simil-Eval: Adding New Metrics and Datasets

This guide provides comprehensive instructions for extending the **Simil-Eval** framework by adding new **similarity metrics**, **datasets**, and **tasks**. Whether you're incorporating a novel evaluation metric or adding support for a new dataset/language, this document walks you through every step.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Adding New Similarity Metrics](#2-adding-new-similarity-metrics)
3. [Adding New Datasets (Similarity Tasks)](#3-adding-new-datasets-similarity-tasks)
4. [Adding New Surprisal Tasks](#4-adding-new-surprisal-tasks)
5. [Testing Your Changes](#5-testing-your-changes)
6. [Common Patterns and Best Practices](#6-common-patterns-and-best-practices)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Architecture Overview

### Project Structure

```
simil-eval/
├── core/
│   ├── SimilarityTask.py          # Base class for similarity-based tasks
│   ├── SurprisalTask.py           # Base class for surprisal-based tasks
│   ├── sim_steps.py               # Similarity evaluation pipeline
│   ├── sur_steps.py               # Surprisal evaluation pipeline
│   └── tasks/                     # Individual task implementations
│       ├── Openbookqa.py
│       ├── Belebele.py
│       ├── Summarization.py
│       └── ... (other tasks)
├── utils/
│   ├── metrics.py                 # Metric implementations
│   ├── surprisal.py               # Surprisal computation
│   └── moverscore_v2.py           # MoverScore implementation
├── configs/
│   ├── bert_models.yaml           # BERT models by language
│   └── tasks_ubication.yaml       # Dataset locations
├── eval_similarity.py             # Main similarity evaluation script
├── eval_surprisal.py              # Main surprisal evaluation script
└── ...
```

### Key Concepts

- **SimilarityTask**: Abstract base class for tasks that require text generation and similarity comparison (e.g., QA, summarization)
- **SurprisalTask**: Abstract base class for tasks that evaluate text acceptability (e.g., grammatical acceptability)
- **Metrics**: Functions that compute similarity or surprisal scores between texts
- **Dataset Configuration**: YAML files that map dataset names and languages to Hugging Face Dataset IDs

---

## 2. Adding New Similarity Metrics

### Step 1: Implement the Metric Function in `utils/metrics.py`

Add your new metric function to [utils/metrics.py](utils/metrics.py). The function should accept the necessary inputs and return a numerical score.

Let's say you want to add a **simple token overlap metric** (Jaccard similarity):

```python
def jaccard_similarity(sentence1, sentence2):
    """
    Compute Jaccard similarity (token overlap) between two sentences.
    
    Args:
        sentence1 (str): First sentence
        sentence2 (str): Second sentence
    
    Returns:
        float: Jaccard similarity score (0-1)
    """
    tokens1 = set(sentence1.lower().split())
    tokens2 = set(sentence2.lower().split())
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)
```

### Step 2: Update `sim_steps.py` to Use Your Metric

In [core/sim_steps.py](core/sim_steps.py), locate the `compute_sentence_similarity()` function and add your metric:

```python
def compute_sentence_similarity(task:SimilarityTask, metric, tokenizer, model, sentence1, sentence2):
    if metric == "cosine":
        return sim_metrics.cosine_score(tokenizer, model, sentence1, sentence2)
    elif metric == "moverscore":
        bert_model = bertmodels_yaml[task.lang]
        return sim_metrics.mover_score(bert_model, sentence1, sentence2)
    elif metric == "jaccard":                          # ← NEW
        return sim_metrics.jaccard_similarity(sentence1, sentence2)  # ← NEW
    else:
        raise NotImplementedError
```

### Step 3: Handle Corpus-Level Metrics (Optional)

Some metrics like **BERTScore** operate on entire corpora rather than individual sentence pairs. If your metric is corpus-level, also update `compute_corpus_similarity()`:

```python
def compute_corpus_similarity(task:SimilarityTask, metric, generations, references):
    if metric == "bertscore":
        # ... existing BERTScore code ...
    elif metric == "your_corpus_metric":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        results = sim_metrics.your_corpus_metric(generations, references)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        return format_results(results)
    else:
        raise NotImplementedError
```

### Step 4: Register Your Metric in `eval_similarity.py`

In [eval_similarity.py](eval_similarity.py), update the `supported_metrics` list:

```python
if args.evaluate_similarity:
    supported_metrics = ["cosine", "bertscore", "moverscore", "jaccard"]  # ← ADD HERE
    for metric in args.metrics:
        if metric not in supported_metrics:
            raise ValueError(f"Metric {metric} not supported. Choose from {supported_metrics}")
        evaluate_similarity(task, evaluated_model, metric, args.results_file)
```

### Step 5: Test Your Metric

```bash
python3 eval_similarity.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --dataset openbookqa_gl \
  --language gl \
  --metrics jaccard \
  --fewshot_num 0 \
  --create_examples \
  --generate_answers \
  --evaluate_similarity
```

---

## 3. Adding New Datasets (Similarity Tasks)

### Understanding the Task Class

All similarity-based tasks inherit from `SimilarityTask` and must implement these methods:

| Method | Purpose |
|--------|---------|
| `build_prompt()` | Constructs the prompt to send to the LLM |
| `get_correct_option()` | Extracts the correct/reference answer |
| `get_options()` | Returns all available options (for multiple choice) |
| `load_dataset()` | Loads the dataset (usually called automatically) |

### Step 1: Create a New Task Class

Create a new file in [core/tasks/](core/tasks/) following the naming convention. For example, if you're adding QuAC dataset support, create `core/tasks/Quac.py`:

```python
from core.SimilarityTask import SimilarityTask
from datasets import load_dataset

class Quac(SimilarityTask):
    """
    Class for the QuAC (Question Answering in Context) task.
    This is a reading comprehension dataset with follow-up questions.
    """

    def __init__(self, lang, cache, token=""):
        super().__init__(
            "quac",                    # Task name
            lang,                      # Language code
            "ANSWER:",                 # Prompt split marker
            cache,
            token
        )
        self.load_dataset()

    def load_dataset(self):
        """Load the QuAC dataset from Hugging Face."""
        if self.lang == "en":
            self.dataset = load_dataset(
                "quac",
                cache_dir=self.cache
            )["validation"]
        else:
            raise ValueError(f"QuAC language {self.lang} not supported yet")

    def get_correct_option(self, example):
        """
        Extract the correct/reference answer(s).
        QuAC has multiple valid answers.
        
        Args:
            example (dict): Data point from the dataset
        
        Returns:
            str or list: The correct answer(s)
        """
        # QuAC stores answers in example['answers']['texts']
        answers = example['answers']['texts']
        return answers[0] if answers else ""

    def get_options(self, example):
        """
        Get all answer options (for consistency with multiple-choice tasks).
        QuAC is extractive, so we return only the text being questioned.
        
        Args:
            example (dict): Data point from the dataset
        
        Returns:
            list: List of text segments or just the answer
        """
        return example['answers']['texts']

    def build_prompt(self, example, show_answer, show_options=True):
        """
        Build the evaluation prompt for QuAC.
        
        Args:
            example (dict): Data point from the dataset
            show_answer (bool): Include the reference answer
            show_options (bool): Include context (required for extractive QA)
        
        Returns:
            str: The formatted prompt
        """
        context = example['context']
        question = example['question']
        
        prompt = f"""CONTEXT: {context}

QUESTION: {question}

ANSWER:"""
        
        if show_answer:
            correct_answer = self.get_correct_option(example)
            prompt += f" {correct_answer}\n"
        
        return prompt
```

### Step 2: Register Your Dataset in `configs/tasks_ubication.yaml`

Add your dataset configuration to [configs/tasks_ubication.yaml](configs/tasks_ubication.yaml):

**Add your new dataset:**
```yaml
quac:
  - en: quac                          # ← Hugging Face dataset ID
  - es: quac                          # if available
```

**Important Notes:**
- Format: `task_name: - language: huggingface_dataset_id`
- Some datasets require a tuple with dataset name and config: `[dataset_name, config_name]`
- Example: `[facebook/belebele, eng_Latn]` for Belebele with language configs

### Step 3: Update `eval_similarity.py` to Import Your Task

In [eval_similarity.py](eval_similarity.py), add the import:

```python
from core.tasks import Belebele, Openbookqa, Summarization, Xstorycloze, \
                       Truthfulqa_mc1, Veritasqa_mc1, Quac  # ← ADD HERE
```

### Step 4: Add Task Instantiation Logic

In [eval_similarity.py](eval_similarity.py), add the task instantiation in the argument parsing section:

```python
if args.dataset == "belebele":
    task = Belebele.Belebele(lang=args.language, cache=args.cache)

# ... existing code ...

elif args.dataset == "quac":                           # ← ADD NEW TASK
    task = Quac.Quac(lang=args.language, cache=args.cache, token=args.token)

else:
    exit("Task not supported. Currently implemented tasks are [Belebele, OpenBookQA, QuAC, ...]")
```

---

## 4. Adding New Surprisal Tasks

Surprisal tasks evaluate text acceptability without generation. They're used for grammatical acceptability, linguistic preferences, etc.

### Step 1: Create a New Surprisal Task Class

Create a file in [core/tasks/](core/tasks/) (e.g., `Gramtest.py`):

```python
from core.SurprisalTask import SurprisalTask
from datasets import load_dataset

class Gramtest(SurprisalTask):
    """
    Grammatical acceptability task for testing language model's
    preference for grammatically correct vs. incorrect sentences.
    """

    def __init__(self, lang, cache_dir):
        super().__init__(lang, cache_dir)
        self.dataset_good = []
        self.dataset_bad = []
        self.load_evaluation_dataset()

    def load_evaluation_dataset(self):
        """
        Load the dataset with grammatically correct and incorrect examples.
        Different languages may require different loading logic.
        """
        if self.lang == "en":
            self._load_englishgramtest()
        elif self.lang == "gl":
            self._load_galiciangramtest()
        else:
            raise ValueError(f"Gramtest language {self.lang} not supported")

    def _load_englishgramtest(self):
        """Load English grammatical acceptability test set."""
        dataset = load_dataset(
            "your_org/englishgramtest",
            cache_dir=self.cache_dir
        )["test"]
        
        self.dataset_good = [
            item['sentence'] for item in dataset 
            if item['grammatical'] == True
        ]
        self.dataset_bad = [
            item['sentence'] for item in dataset 
            if item['grammatical'] == False
        ]

    def _load_galiciangramtest(self):
        """Load Galician grammatical acceptability test set."""
        dataset = load_dataset(
            "proxectonos/galician_gramtest",
            cache_dir=self.cache_dir
        )["test"]
        
        self.dataset_good = [
            item['sentence'] for item in dataset 
            if item['is_correct'] == True
        ]
        self.dataset_bad = [
            item['sentence'] for item in dataset 
            if item['is_correct'] == False
        ]
```

### Step 2: Update `eval_surprisal.py`

In [eval_surprisal.py](eval_surprisal.py), add the import and instantiation:

```python
from core.tasks import Cola, Gramtest  # ← ADD YOUR TASK

# In argument parsing section:
if args.dataset == "cola":
    task = Cola.Cola(lang=args.lang, cache_dir=args.cache)
elif args.dataset == "gramtest":           # ← ADD HERE
    task = Gramtest.Gramtest(lang=args.lang, cache_dir=args.cache)
else:
    exit("Task not supported")
```

### Step 3: Test Your Surprisal Task

```bash
python3 eval_surprisal.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --dataset gramtest \
  --lang en \
  --cache ./cache
```

---

## 5. Testing Your Changes

### Unit Testing Your Metric

Create a simple test script to verify your metric works correctly:

```python
# test_metric.py
from utils import metrics

# Test sentence pairs
sentence1 = "The cat sat on the mat"
sentence2 = "The cat was sitting on the mat"

# Test your metric
score = metrics.jaccard_similarity(sentence1, sentence2)
print(f"Jaccard similarity: {score}")

# Verify output is in expected range (0-1 for normalized metrics)
assert 0 <= score <= 1, f"Score {score} out of range"

# Test edge cases
print("Testing edge cases...")
print(f"Empty strings: {metrics.jaccard_similarity('', '')}")
print(f"Identical strings: {metrics.jaccard_similarity('test', 'test')}")
print(f"No overlap: {metrics.jaccard_similarity('abc', 'xyz')}")
```

### Integration Testing

Test the entire pipeline:

```bash
# 1. Create test examples
python3 eval_similarity.py \
  --dataset openbookqa \
  --language en \
  --cache ./cache \
  --create_examples \
  --fewshot_num 1

# 2. Generate answers (small subset for testing)
python3 eval_similarity.py \
  --dataset openbookqa \
  --language en \
  --cache ./cache \
  --generate_completions \
  --results_file ./test_results.json

# 3. Evaluate with your new metric
python3 eval_similarity.py \
  --dataset openbookqa \
  --language en \
  --cache ./cache \
  --evaluate_similarity \
  --metrics jaccard
```

### Verify Dataset Loading

```python
# test_dataset.py
from core.tasks import Quac

# Test dataset loading
task = Quac.Quac(lang="en", cache="./cache")

# Verify dataset is loaded
print(f"Dataset loaded: {task.dataset is not None}")
print(f"Number of examples: {len(task.dataset)}")

# Test prompt building
if len(task.dataset) > 0:
    example = task.dataset[0]
    prompt = task.build_prompt(example, show_answer=True)
    print(f"Sample prompt:\n{prompt}")
```
