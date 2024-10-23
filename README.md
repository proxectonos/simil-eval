# Framework de similaridade
Ferramenta para avaliar LLMs empleando medidas de similaridade entre embeddings.

## Setup

1.  Instalar o entorno de avaliación (pendente)
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
4.  Lanzar mediante ```sbatch`` ás colas do Cesga:
   -    **CoLA**: ```sbatch launch_CoLA```: Obtéñense resultados para as versións do CoLA en galego, catalán, español e inglés.
   -    **Calame**: ```sbatch launch_Calame.sh```: Obtéñense resultados para Calame-PT (pendente Calame-GL).

## Características da ferramenta

### Métricas

- **Coseno**:
- **MoverScore**:
- **BertScore**:
- **Surprisal**:

### Datasets
- **Belebele**:
- **OpenBookQA**:
- **Calame**:
- **Cola**:

### Resumo

|            |       Coseno       |     MoverScore     |      BertScore     |      surprisal     |
|------------|:------------------:|:------------------:|:------------------:|:------------------:|
| OpenBookQA | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
|  Belebele  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
|   Calame   |                    |                    |                    | :heavy_check_mark: |
|    Cola    |                    |                    |                    | :heavy_check_mark: |