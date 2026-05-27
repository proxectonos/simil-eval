# Developer Guide

Reference documentation for developers maintaining and extending the Simil-Eval project.

---

## 1. Architecture Overview


### Design Principles

- **Task Abstraction**: Similarity and surprisal tasks are defined through abstract base classes (`SimilarityTask`, `SurprisalTask`), allowing new datasets to be added by implementing required methods
- **Metric Decoupling**: Metrics are independent functions in `utils/metrics.py`, separate from evaluation logic
- **Configuration-Driven**: Dataset and model mappings are externalized in YAML files, avoiding hard-coded references
- **Two-Stage Pipeline (Similarity)**: Examples → Generation → Evaluation. Each stage can run independently, enabling iterative development

### Evaluation Approaches

**Similarity-Based**: Generates text using an LLM and compares output to reference answers using embedding-based metrics. Useful for QA, summarization, and generation tasks.

**Surprisal-Based**: Evaluates model's linguistic competence without generation by comparing probability assigned to correct vs. incorrect sentences. Efficient for acceptability judgments.

---

## 2. Project Structure

```
simil-eval/
├── core/                          # Core evaluation logic
│   ├── SimilarityTask.py          # Abstract base for similarity tasks
│   ├── SurprisalTask.py           # Abstract base for surprisal tasks
│   ├── sim_steps.py               # Similarity evaluation pipeline
│   ├── sur_steps.py               # Surprisal evaluation pipeline
│   └── tasks/                     # Task implementations (Openbookqa.py, Cola.py, etc.)
│
├── utils/                         # Utility functions
│   ├── metrics.py                 # Metric implementations (cosine, bertscore, moverscore)
│   ├── surprisal.py               # Surprisal computation
│   └── moverscore_v2.py           # MoverScore metric
│
├── configs/                       # Configuration files (external references)
│   ├── bert_models.yaml           # Language → BERT model mappings
│   └── tasks_ubication.yaml       # Task/language → HuggingFace dataset mappings
│
├── eval_similarity.py             # CLI entry point for similarity evaluation
├── eval_surprisal.py              # CLI entry point for surprisal evaluation
│
├── export/                        # Excel export utilities
├── launchers/                     # SLURM job scripts
├── cache/                         # Downloaded models/datasets
└── generated_files/               # Output: examples, results, metrics
```

---

## 3. Core Abstractions

### SimilarityTask Base Class
Defines the interface for similarity-based tasks. Each task must implement:

- **`build_prompt(example, show_answer, show_options)`**: Formats dataset examples into prompts for the model
- **`get_correct_option(example)`**: Extracts the reference/correct answer
- **`get_options(example)`**: Returns available options (for multiple choice tasks)
- **`load_dataset()`**: Loads the dataset from HuggingFace (automatic)

Example: `Openbookqa.py` implements these for multiple-choice QA; `Belebele.py` for reading comprehension.

### SurprisalTask Base Class
Simpler interface for acceptability-based tasks:

- **`load_evaluation_dataset()`**: Must populate `self.dataset_good` and `self.dataset_bad`

Example: `Cola.py` loads grammatically correct/incorrect sentences.

### Metrics
Functions in `utils/metrics.py` compute similarity between generated and reference text:

- **Sentence-level**: `cosine_score()`, `mover_score()` - compare single pairs
- **Corpus-level**: `bert_score()` - compute aggregate scores across many pairs

---

## 4. Data Flow

### Similarity Evaluation Pipeline

```
Dataset → build_prompt() → LLM Prompt
                            ↓
                          Generation
                            ↓
                         CSV Results
                            ↓
                       metric(generated, reference)
                            ↓
                        Metrics (CSV)
```

**Key**: Each stage produces intermediate outputs (CSV files) that can be inspected or reused.

### Surprisal Evaluation Pipeline

```
dataset_good / dataset_bad → compute_surprisal(text)
                            ↓
                          S(correct) vs S(incorrect)
                            ↓
                          difsur score
```

---

## 5. Configuration Management

All external references are in YAML files to avoid code changes:

### `configs/tasks_ubication.yaml`
Maps task/language combinations to HuggingFace dataset IDs:
```yaml
openbookqa:
  - gl: proxectonos/openbookqa_gl
  - en: cnut1648/openbookqa_retrieved_by_colbert
```

When adding a language to a task, add one line here.

### `configs/bert_models.yaml`
Maps language codes to BERT models for embedding metrics:
```yaml
gl: marcosgg/bert-base-gl-cased
en: google-bert/bert-base-uncased
```

When supporting a new language, ensure a model exists here.