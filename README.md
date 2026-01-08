# Simil-Eval: A Multilingual Toolkit for Evaluating LLMs

**Simil-Eval** is a comprehensive toolkit for evaluating Large Language Models (LLMs) using **interpretability-based metrics**. Unlike traditional evaluation suites that rely solely on exact matching or accuracy, Simil-Eval incorporates **embedding-based similarity** and **surprisal (perplexity)** to provide deeper insights into model behavior—particularly for **low-resource languages**.

This repository accompanies the paper [Continued Pretraining and Interpretability-Based Evaluation for Low-Resource Languages: A Galician Case Study](https://aclanthology.org/2025.findings-acl.240/) (Rodríguez et al., Findings 2025)

---

## Table of Contents

1. [Methodology & Metrics](#1-methodology--metrics)  
2. [Setup](#2-setup)  
3. [Quick Start (Local Usage)](#3-quick-start-local-usage)  
4. [Cluster Deployment (SLURM)](#4-cluster-deployment-slurm)  
5. [Available Datasets & Supported Metrics](#5-available-datasets--supported-metrics)  
6. [Exporting Results](#6-exporting-results)  
7. [Citation](#7-citation)

---

## 1. Methodology & Metrics

Simil-Eval evaluates LLMs from two complementary perspectives:

- **Surprisal** → intrinsic knowledge and linguistic competence (no generation required)
- **Similarity** → semantic quality of generated text

### A. Surprisal Metrics

Surprisal-based metrics evaluate how *expected* a given text is according to a language model.

#### Surprisal ($S(x)$) ([**Code**](https://github.com/kanishkamisra/minicons),[**Paper**](https://arxiv.org/pdf/2203.13112))

Surprisal measures the negative log-probability of a token sequence. This metric is especially useful for **linguistic acceptability** and **commonsense reasoning** tasks.

*Interpretation*:
  - Lower values → the model finds the text natural or expected
  - Higher values → the model is unfamiliar with the language, structure, or facts


#### Difsur (Differential Surprisal)

A novel metric introduced in Simil-Eval to explicitly compare correct vs. incorrect alternatives:

$$\text{difsur} = \frac{S(x_{na}) - S(x_a)}{\max\{S(x_a), S(x_{na})\}} \times 100$$

where:
- $S(x_a)$: surprisal of the acceptable / correct text
- $S(x_{na})$: surprisal of the non-acceptable / incorrect text

*Interpretation*:
  - Higher values indicate a stronger model preference for the correct option
  - Values near zero suggest weak discrimination

---

### B. Similarity Metrics

Similarity metrics evaluate how close a model’s generated output is to a reference answer in semantic space.

- **Cosine Similarity**: Computes cosine similarity between sentence embeddings of the model generation and the reference answer. This metric is fast and model-agnostic.

- **MoverScore** ([**Code**](https://github.com/AIPHES/emnlp19-moverscore),[**Paper**](https://arxiv.org/pdf/1909.02622)): Uses embeddings from a BERT model to calculate the *effort* required to transform one text into another. Lower effort means higher similarity.

- **BERTScore** ([**Code**](https://github.com/Tiiiger/bert_score),[**Paper**](https://arxiv.org/pdf/1904.09675)): Uses embeddings from a BERT model to compute a refined version of Cosine Similarity.

---

## 2. Setup

> **Note**  
> Simil-Eval is extensively tested on **SLURM clusters**, but also runs efficiently on a single local machine.

### Prerequisites

- Python **3.9+**
- A working `pip` installation
- CUDA-enabled GPU recommended (optional but strongly advised)

### Installation

Clone the repository and install dependencies in a python environment:

```bash
git clone https://github.com/proxectonos/simil-eval
sh install.sh
```

### Environment variables configuration

Create a file at `./configs/.env` with the following structure:

```bash
HF_TOKEN=your_huggingface_token_here
CACHE_DIR=./cache
```

- `HF_TOKEN`: Hugging Face access token (required for gated models)
- `CACHE_DIR`: directory for model and dataset caching

---

## 3. Quick Start (Local Usage)

> [!NOTE]  
> This tool is extensively tested in a cluster managed by SLURM. It can be used in other environments, but some modifications may be necessary.

### A. Similarity Evaluation (Generation)

Example: evaluating a `Llama2-7b` on the Galician OpenBookQA using five few-shot examples.

```bash
source eval_env/bin/activate
python3 eval_similarity.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --dataset openbookqa_gl \
  --language gl \
  --metrics cosine bertscore \
  --fewshot_num 5 \
  --create_examples \
  --generate_answers \
  --evaluate_similarity \
  --results_file ./results/llama_gl_eval.json
```

#### Common Arguments

| Argument | Description |
|--------|-------------|
| `--model` | Hugging Face model ID or local path |
| `--dataset` | Dataset identifier |
| `--language` | Language code (`gl`, `es`, `en`, `pt`, `cat`) |
| `--metrics` | `cosine`, `moverscore`, `bertscore` |
| `--fewshot_num` | Number of in-context examples (0 = zero-shot) |
| `--create_examples` | Construct few-shot prompts |
| `--generate_answers` | Generate responses for the selected datasets |
| `--evaluate_similarity` | Computes similarity metrics |
| `--results_file` | Output JSON file |

---

### B. Surprisal Evaluation

Example: evaluating a `Llama2-7b` on the Catalan Cola version.

```bash
source eval_env/bin/activate
python3 eval_surprisal.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --dataset catcola \
  --lang cat \
  --cache ./cache
```

---

## 4. Cluster Deployment (SLURM)

Simil-Eval supports large-scale evaluations on SLURM-managed clusters.

### Similarity Jobs

1. Navigate to `./launchers/`
2. Edit `execute_eval_similarity.sh`:
   - `MODELS`: Hugging Face IDs or local checkpoints
   - `DATASETS`: dataset identifiers
   - `LANGUAGES`: `gl`, `cat`, `es`, `en`, `pt`
3. Adjust `#SBATCH` directives (GPUs, memory, time) in the `launch_sim_eval.sh` file.
4. Launch:

```bash
sh execute_eval_similarity.sh
```

### Surprisal Jobs

1. Navigate to `./launchers/`
2. Edit `execute_eval_surprisal.sh` to specify `MODELS`
3. Adjust `#SBATCH` directives (GPUs, memory, time) in the `launch_sur_eval.sh` file. 
4. Launch:

```bash
sh execute_eval_surprisal.sh
```

---

## 5. Available Tasks, Datasets & Supported Metrics

### Tasks and Available Datasets

- **Multiple Choice QA**:
  - [*OpenBookQA*](https://aclanthology.org/D18-1260/): Four possible answers. A question is asked, and the correct answer must be selected.
  - [*VeritasQA*](https://aclanthology.org/2025.coling-main.366/): Between 4 and 10 possible options. A question is asked and the correct option must be chosen, but there are multiple correct options possible. The task is similar to the original *mc1* showed in its paper.
  - [*TruthfulQA*](https://aclanthology.org/2022.acl-long.229/): Similar to VeritasQA but focused in USA facts. The task is similar to the original *mc1* showed in its paper.
- **Reading Understanding**
  - [*Belebele*](https://aclanthology.org/2024.acl-long.44/): A context is provided with diverse information, followed by a question with 4 options where the correct answer can be deduced from the context.
  - [*XStoryCloze*](https://aclanthology.org/2022.emnlp-main.616/): A context is provided with diverse information, followed for two options that can complete the context. The model has to choose the *logical* option to continue the text.
- **Linguistic Acceptability**:  
  - [*CoLA*](https://aclanthology.org/Q19-1040/): Contains sentences labeled as linguistically acceptable (1) or unacceptable (0). This allows for studying when a model is more likely to generate acceptable or unacceptable texts by comparing the probabilities it assigns to each type of sentence.
- **Generative Capabilities**:
  - [*Calame*](https://aclanthology.org/2024.propor-1.45/): A text fragment is provided, and the task is to complete the last word. The dataset is designed so that the last word should be unique, and the goal is to check whether the word generated by the model matches the reference word in the dataset for each fragment.
- **Physical commonsense reasoning**:  
  - [*Global PIQA*](https://arxiv.org/abs/2510.24081): Each example consists of two candidate solutions, one correct and one incorrect. Determining the correct solution is designed to require physical commonsense reasoning, although we allow for fairly flexible definitions of physical commonsense (e.g. knowledge of physical properties of objects, affordances, physical and temporal relations, and everyday activities).


### Supported Metrics by Dataset

|             | Cosine Similarity  | MoverScore         | BertScore          | Surprisal          |
|-------------|--------------------|--------------------|--------------------|--------------------|
| OpenBookQA  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| Belebele    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| CoLA        |                    |                    |                    | :heavy_check_mark: |
| Calame      |                    |                    |                    | :heavy_check_mark: |
| VeritasQA   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| XStoryCloze | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| TruthfulQA  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    | 
| GlobalPIQA  |                    |                    |                    | :heavy_check_mark: |

---

### Datasets by Language

|             | Galician                                                                                                | English                                                                                                  | Catalan                                                                                                  | Spanish                                                                                                            | Portuguese                                                                                                         |
|-------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| OpenBookQA  | [openbookqa_gl](https://huggingface.co/datasets/proxectonos/openbookqa_gl)                              | [openbookqa](https://huggingface.co/datasets/cnut1648/openbookqa_retrieved_by_colbert)                   | [openbookqa_ca](https://huggingface.co/datasets/projecte-aina/openbookqa_ca)                             | [openbookqa_es](https://huggingface.co/datasets/BSC-LT/openbookqa-es)                                              | Private                                                                                                            |
| Belebele    | [belebele_gl](https://huggingface.co/datasets/proxectonos/belebele_gl)                                  | [belebele_eng_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/eng_Latn)                   | [belebele_cat_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/cat_Latn)                   | [belebele_spa_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/spa_Latn)                             | [belebele_por_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/por_Latn)                             |
| CoLA        | [galcola](https://huggingface.co/datasets/proxectonos/galcola)                                          | [glue_cola](https://huggingface.co/datasets/nyu-mll/glue/viewer/cola)                                    | [CatCoLA](https://huggingface.co/datasets/nbel/CatCoLA)                                                  | [EsCoLA](https://huggingface.co/datasets/nbel/EsCoLA)                                                              |                                                                                                                    |
| Calame      | [calame-gl](https://github.com/proxectonos/calame-gl/tree/main)                                         |                                                                                                          |                                                                                                          |                                                                                                                    | [calame-pt](https://huggingface.co/datasets/NOVA-vision-language/calame-pt)                                        |
| VeritasQA   | [veritasqa_gl](https://huggingface.co/datasets/projecte-aina/veritasQA/viewer/default/gl)               | [veritasqa_en](https://huggingface.co/datasets/projecte-aina/veritasQA/viewer/default/en)                | [veritasqa_ca](https://huggingface.co/datasets/projecte-aina/veritasQA/viewer/default/ca)                | [veritasqa_es](https://huggingface.co/datasets/projecte-aina/veritasQA/viewer/default/es)                          |                                                                                                                    |
| XStoryCloze | [xtorycloze_gl](https://huggingface.co/datasets/proxectonos/xstorycloze_gl)                             | [xstory_cloze_en](https://huggingface.co/datasets/juletxara/xstory_cloze/viewer/en)                      | [xstorycloze_ca](https://huggingface.co/datasets/projecte-aina/xstorycloze_ca)                           | [xstory_cloze_es](https://huggingface.co/datasets/juletxara/xstory_cloze/viewer/es)                                | [XStoryCloze_pt](https://huggingface.co/datasets/proxectonos/XStoryCloze_pt)                                       |
| TruthfulQA  | [truthfulqa_gl_gen](https://huggingface.co/datasets/proxectonos/truthfulqa_gl/viewer/generation)        | [truthful_qa_gen](https://huggingface.co/datasets/truthfulqa/truthful_qa/viewer/generation/)             |                                                                                                          |                                                                                                                    |                                                                                                                    |
| Global PIQA | [global-piqa_gl](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel/viewer/glg_latn) | [global-piqa_eng](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel/viewer/eng_latn) | [global-piqa_cat](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel/viewer/cat_latn) | [global-piqa_spa-spai](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel/viewer/spa_latn_spai) | [global-piqa_por-port](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel/viewer/por_latn_port) |
---

## 6. Exporting Results

Convert raw JSON outputs into Excel summaries.

1. Load the evaluation environment.
2. Execute the following commands, changing the `RESULTS_DIR` and `OUTPUT_DIR` variables in each file:
```bash
cd export/
python export_similarity_to_excel.py
python export_surprisal_to_excel.py
```

Outputs (located in `$OUTPUT_DIR/`):
- `similarity_summary.xlsx`
- `surprisal_summary.xlsx`

## 7. Citation

If you use Simil-Eval or the Galician datasets, please cite:

```bibtex
@inproceedings{rodriguez-etal-2025-continued,
    title = "Continued Pretraining and Interpretability-Based Evaluation for Low-Resource Languages: A {G}alician Case Study",
    author = "Rodr{\'i}guez, Pablo  and
      Su{\'a}rez, Silvia Paniagua  and
      Gamallo, Pablo  and
      Docio, Susana Sotelo",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.240/",
    doi = "10.18653/v1/2025.findings-acl.240",
    pages = "4622--4637",
    ISBN = "979-8-89176-256-5",
    abstract = "Recent advances in Large Language Models (LLMs) have led to remarkable improvements in language understanding and text generation. However, challenges remain in enhancing their performance for underrepresented languages, ensuring continual learning without catastrophic forgetting, and developing robust evaluation methodologies. This work addresses these issues by investigating the impact of Continued Pretraining (CPT) on multilingual models and proposing a comprehensive evaluation framework for LLMs, focusing on the case of Galician language. Our first contribution explores CPT strategies for languages with limited representation in multilingual models. We analyze how CPT with Galician corpora improves text generation while assessing the trade-offs between linguistic enrichment and task-solving capabilities. Our findings show that CPT with small, high-quality corpora and diverse instructions enhances both task performance and linguistic quality. Our second contribution is a structured evaluation framework based on distinguishing task-based and language-based assessments, leveraging existing and newly developed benchmarks for Galician. Additionally, we contribute new Galician LLMs, datasets for evaluation and instructions, and an evaluation framework."
}
```

---
