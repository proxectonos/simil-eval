# similaridade-eval
Tool for evaluating LLMs using similarity measures between embeddings and surprisals in sentences.

> [!NOTE]  
> This tool is extensively tested in a cluster managed by SLURM. It can be used in other environments, but some modifications may be necessary.

## Setup

1. Install the evaluation environment and create the necessary auxiliary folders:
    ```sh install.sh```
2. Edit the environment variables in the file ```./configs/.env```.

## Perform Evaluations in SLURM clusters

### Similarity Evaluations
1. Load the evaluation environment.
2. Navigate to the ```./launchers``` folder.
3. Update the following fields in the execution file ```execute_eval_similarity.es```:
    - **MODELS**: Models to evaluate. These can be references from HuggingFace or local paths.
    - **DATASETS**: Datasets to evaluate. Currently supports "openbookqa" and "belebele."
    - **LANGUAGES**: Languages of the dataset to evaluate. Currently available: gl, cat, es, en, pt.
    - **FEWSHOT_NUM**: Number of few-shot examples. To run evaluations without few-shot, set this to 0.
4. Run the script using ```sh execute_eval_similarity.sh```, which will launch processes in the Cesga queues (one for each dataset/model/language combination).

### Surprisal Evaluations

1. Load the evaluation environment.
2. Navigate to the ```./launchers``` folder.
3. Update the following fields in the execution file ```execute_eval_surprisal.sh```:
    - **MODELS**: Models to evaluate. These can be references from HuggingFace or local paths.
4. Run the script using ``sh execute_eval_surprisal``, which will launch processes in the Cesga queues (one for each model).

## Perform evaluations in own computer/server

### Similarity Evaluations
1. Load the evaluation environment.
2. Execute the following command, using the apropriate parameters:
```
   python3 eval_similarity.py \
    --dataset $DATASET \
    --cache $CACHE_DIR \
    --token $TOKEN_HF \
    --model $MODEL \
    --language $LANGUAGE \
    --evaluate_similarity \
    --create_examples \
    --fewshot_num $FEWSHOT_NUM \
    --show_options $SHOW_OPTIONS \
    --examples_file $EXAMPLES_FILE \
    --generate_answers \
    --results_file $RESULTS_FILE \
    --evaluate_similarity \
    --metrics cosine moverscore bertscore
```

### Surprisal Evaluations in own computer/server
1. Load the evaluation environment.
2. Execute the following command, using the apropriate parameters:
```
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset $DATASET --lang $LANG --token $HF_TOKEN
```

## Tool Features

### Metrics

- **Cosine Similarity**: Calculates the cosine similarity between the embeddings of the last layer associated with two text fragments, e.g., between a model’s generation and a reference text fragment.
- **MoverScore** \[[*code*](https://github.com/AIPHES/emnlp19-moverscore), [*paper*](https://arxiv.org/pdf/1909.02622)\]: Uses embeddings from a BERT model to calculate the *effort* required to transform one text into another. For example, it measures how difficult it is to turn a model's generation into a reference text. The lower the *effort*, the higher the similarity between texts.
- **BertScore** \[[*code*](https://github.com/Tiiiger/bert_score), [*paper*](https://arxiv.org/pdf/1904.09675)\]: Uses embeddings from a BERT model to compute a refined version of Cosine Similarity.
- **Surprisal** \[[code](https://github.com/kanishkamisra/minicons), [*paper*](https://arxiv.org/pdf/2203.13112)\]: Measures the *surprise* a model experiences when encountering a token or set of tokens. High values indicate that the model is unaccustomed to generating such tokens. This metric can be used to compare the knowledge of different models in the same language: models with lower surprisal values are theoretically better at the language than those with higher values (which are more *surprised* by the text).

### Tasks and Available Datasets

- **Multiple Choice QA**:
  - *OpenBookQA*: Four possible answers. A question is asked, and the correct answer must be selected.
  - *Belebele*: Four possible answers. A context is provided with diverse information, followed by a question with options where the correct answer can be deduced from the context.
  - *VeritasQA*: Between 4 and 10 possible options. A question is asked and the correct option must be chosen, but there are multiple correct options possible.
- **Linguistic Acceptability**:  
  - *CoLA*: Contains sentences labeled as linguistically acceptable (1) or unacceptable (0). This allows for studying when a model is more likely to generate acceptable or unacceptable texts by comparing the probabilities it assigns to each type of sentence.
- **Generative Capabilities**:
  - *Calame*: A text fragment is provided, and the task is to complete the last word. The dataset is designed so that the last word should be unique, and the goal is to check whether the word generated by the model matches the reference word in the dataset for each fragment.

### Metrics by Dataset

|            |       Cosine Similarity       |     MoverScore     |      BertScore     |      Surprisal     |
|------------|:-----------------------------:|:------------------:|:------------------:|:------------------:|
| OpenBookQA | :heavy_check_mark:            | :heavy_check_mark: | :heavy_check_mark: |                    |
|  Belebele  | :heavy_check_mark:            | :heavy_check_mark: | :heavy_check_mark: |                    |
|   CoLA     |                               |                    |                    | :heavy_check_mark: |
|   Calame   |                               |                    |                    | :heavy_check_mark: |
| VeritasQA  | :heavy_check_mark: | :heavy_check_mark: |                    |                    |

### Datasets by Language

|            | Galician                                                                                   | English                                                                                    | Catalan                                                                                  | Spanish                                                                                   | Portuguese                                                                              |
|------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| OpenBookQA | [openbookqa_gl](https://huggingface.co/datasets/proxectonos/openbookqa_gl)                | [openbookqa](https://huggingface.co/datasets/cnut1648/openbookqa_retrieved_by_colbert)    | [openbookqa_ca](https://huggingface.co/datasets/projecte-aina/openbookqa_ca)              | [openbookqa_es](https://huggingface.co/datasets/BSC-LT/openbookqa-es)                     | Private                                                                                |
| Belebele   | [belebele_gl](https://huggingface.co/datasets/proxectonos/belebele_gl)                    | [belebele_eng_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/eng_Latn)    | [belebele_cat_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/cat_Latn)    | [belebele_spa_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/spa_Latn)    | [belebele_por_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/por_Latn) |
| CoLA       | [galcola](https://huggingface.co/datasets/proxectonos/galcola)                            | [glue_cola](https://huggingface.co/datasets/nyu-mll/glue/viewer/cola)                     | [CatCoLA](https://huggingface.co/datasets/nbel/CatCoLA)                                   | [EsCoLA](https://huggingface.co/datasets/nbel/EsCoLA)                                     |                                                                                        |
| Calame     | In process                                                                                |                                                                                           |                                                                                           |                                                                                           | [calame-pt](https://huggingface.co/datasets/NOVA-vision-language/calame-pt)            |
| VeritasQA  | [veritasqa_gl](https://huggingface.co/datasets/projecte-aina/veritasQA/viewer/default/gl) | [veritasqa_en](https://huggingface.co/datasets/projecte-aina/veritasQA/viewer/default/en) | [veritasqa_ca](https://huggingface.co/datasets/projecte-aina/veritasQA/viewer/default/ca) | [veritasqa_es](https://huggingface.co/datasets/projecte-aina/veritasQA/viewer/default/es) |    
