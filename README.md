# Framework de similaridade
Ferramenta para avaliar LLMs empleando medidas de similaridade entre embeddings.

> [!NOTE]  
> Agora mesmo a feramenta está pensada para utilizarse no Cesga. Próximamente haberá versión para o servidor.

## Setup

1.  Instalar o entorno de avaliación (python 3.9)
    1.  ```python -m venv eval_env```
    2.  ```pip install -R requirements.txt```
2.  Modificar no ficheiro ```./configs/.env``` as variables de entorno.

## Cómo facer avaliacións de coseno/moverscore/bertscore (datasets QA)

1.  Cargar o entorno de avaliación
2.  Moverse á carpeta ./launchers
3.  Modificar no lanzador ```launch_similarity.es``` os seguintes campos:
    -    **MODELS**: Modelos a avaliar. Poden ser referencias a modelos de HuggingFace ou rutas en local.
    -    **DATASETS**: Datasets sobre os que realizará a avaliación. Actualmente admítese "openbookqa" e "belebele".
    -   **LANGUAGES**: Linguas do dataset a avaliar. Actualmente estñan dispoñibles gl, cat, es, en, pt.
    -    **FEWSHOT_NUM**: Número de exemplos de fewshot. Para facer execucións sen fewshot, poñer 0.
4.  Executar con ```sh launch_similarity.sh```, que lanza os procesos correspondentes ás colas do Cesga (1 por cada triplete dataset/modelo/lingua)

## Cómo facer avaliacións de minicons/surprisal

1.  Cargar o entorno de avaliación.
2.  Moverse á carpeta ./launchers
3.  Modificar a variable **MODELS** no lanzador do dataset correspondente (Calame ou CoLA)
4.  Lanzar mediante ```sbatch``` ás colas do Cesga:
   -    **CoLA**: ```sbatch launch_CoLA```: Obtéñense resultados para as versións do CoLA en galego, catalán, español e inglés.
   -    **Calame**: ```sbatch launch_Calame.sh```: Obtéñense resultados para Calame-PT (pendente Calame-GL).

## Características da ferramenta

### Métricas

- **Coseno**: Calcúlase o coseno entre os embeddings da última capa asociados a dous fragmentos textuais. Por exemplo, o coseno entre unha xeración do modelo e un fragmento de referencia ao cal debería parecerse.
- **MoverScore** \[[*código*](https://github.com/AIPHES/emnlp19-moverscore),[*paper*](https://arxiv.org/pdf/1909.02622)\]: Emprega os embeddings obtidos dun modelo BERT para calcular o *esforzo* de transformar un texto noutro. Por exemplo, mide como de difícil e convertir unha xeración do modelo nun texto de referencia. A menor *esforzo*, maior similitude entre os textos.
- **BertScore** \[[*código*](https://github.com/Tiiiger/bert_score),[*paper*](https://arxiv.org/pdf/1904.09675)\]: Emprega os embeddings obtidos dun modelo BERT para calcular unha versión modificada do Coseno.
- **Surprisal** \[[código](https://github.com/kanishkamisra/minicons),[*paper*](https://arxiv.org/pdf/2203.13112)\]: Mide a *sorpresa* que ten o modelo cando ve un token ou conxunto destes. Se o seu valor é alto significa que o modelo non está acostumado a facer xeracións dese tipo. Polo tanto, pode utilizarse para comparar o coñecemento de diferentes modelos nunha mesma lingua: os que teñan surprisal menores coñecerán teóricamente mellor a lingua que os que as teñan máis altas (que estará máis *sorprendidos* de ver os textos).

### Tarefas e datasets dispoñibles
- **Resposta múltiple (QA)**
  - *OpenBookQA*: 4 opcións posibles. Faise unha pregunta e debe elexirse a opción correcta.
  - *Belebele*: 4 opcións posibles. Proporciónase un contexto con diversa información, e logo faise unha pregunta con opcións onde a resposta correcta pode deducirse do contexto.
- **Aceptabilidade Lingüística**:   
  - *CoLA*: Consta de oracións etiquetadas como aceptables (1) ou non (0) lingüísticamente. Permite estudar cando un modelo é máis propenso a escribir máis textos aceptables ou non, comparando as probabilidades que da a cada tipo de oración.
- **Capacidades xerativas**:
  - *Calame*: Dase un fragmento textual e pídese completar a última palabra. Está construído de tal forma que esta última palabra debería ser única, e mírase se coincide a xerada polo modelo coa que se proporciona de referencia no dataset para cada fragmento.

### Métricas empregadas por dataset

|            |       Coseno       |     MoverScore     |      BertScore     |      Surprisal     |
|------------|:------------------:|:------------------:|:------------------:|:------------------:|
| OpenBookQA | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
|  Belebele  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
|   CoLA     |                    |                    |                    | :heavy_check_mark: |
|   Calame   |                    |                    |                    | :heavy_check_mark: |

### Datasets por lingua

|            |                                   Galego                                   |                                         Inglés                                         |                                         Catalán                                        |                                         Español                                        | Portugués                                                                              |
|------------|:--------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------|
| OpenBookQA | [openbookqa_gl](https://huggingface.co/datasets/proxectonos/openbookqa_gl) | [openbookqa](https://huggingface.co/datasets/cnut1648/openbookqa_retrieved_by_colbert) | [openbookqa_ca](https://huggingface.co/datasets/projecte-aina/openbookqa_ca)           | [openbookqa_es](https://huggingface.co/datasets/BSC-LT/openbookqa-es)                  | Pendente                                                                               |
| Belebele   | [belebele_gl](https://huggingface.co/datasets/proxectonos/belebele_gl)     | [belebele_eng_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/eng_Latn) | [belebele_cat_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/cat_Latn) | [belebele_spa_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/spa_Latn) | [belebele_por_Latn](https://huggingface.co/datasets/facebook/belebele/viewer/por_Latn) |
| CoLA       | [galcola](https://huggingface.co/datasets/proxectonos/galcola)             | [glue_cola](https://huggingface.co/datasets/nyu-mll/glue/viewer/cola)                  | [CatCoLA](https://huggingface.co/datasets/nbel/CatCoLA)                                | [EsCoLA](https://huggingface.co/datasets/nbel/EsCoLA)                                  |                                                                                        |
| Calame     |                                  Pendente                                  |                                                                                        |                                                                                        |                                                                                        | [calame-pt](https://huggingface.co/datasets/NOVA-vision-language/calame-pt)            |

