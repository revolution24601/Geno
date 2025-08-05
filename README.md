# Foundation Model for Joint Reasoning Over Genomic Sequence Syntax and Biological Semantics

This repo contains the code and models for "[Foundation Model for Joint Reasoning Over Genomic Sequence Syntax and Biological Semantics]()".

## Contents
1. [Model](#model)
2. [Results](#results)

## Model
**Geno** is a biologically grounded foundation model that unifies sequence generation, modality translation, and token-level annotation within a single causal modeling framework. A three-phase pretraining curriculum, coupled with domain-informed architectural design and centered on task-conditioned prompting and dual-label supervision, enables bridging of sequence syntax and biological semantics across multiple levels of resolution. This design allows the model to perform both macro-scale regulatory inference and nucleotide-resolution annotation without task-specific architecture changes.

We have released the following checkpoints:

|          Checkpoint           | Description                                                                                                                                |
| :---------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           Geno-Zero           | checkpoint after Phase~III.                                                          |


You can download the weights from:
- [Google Drive](https://drive.google.com/drive/folders/1WaoYB-azdB7oBOHiy7nzO-C4ObcB-H2u?usp=drive_link)

## Results
1. sequence-level tasks
We evaluated our model on a diverse set of genomic tasks drawn from the GUE benchmark suites. These tasks are organized according to the core categories of biological inference, including the identification of the transcription factor (TF) binding site, the classification of the promoter and the core promoter, the recognition of the splice site, the prediction of histone marks and the detection of the long-range enhancer-promoter interaction.
Geno-Zero refers to our model evaluated in zero-shot mode. Tasks are grouped by biological categories. Bolded values indicate the best score among the four models either per row (i.e., per dataset) or in the average row of each task type.

| Task Type                           | Dataset            | Geno-Zero-FT   | NT-500M-FT   | EVO2-FT       | CD-GPT-FT     |
| ----------------------------------- | ------------------ | ----------- | --------- | ---------- | ---------- |
| Splice Site                         | splice reconstruct | **0.8349**  | 0.6901    | 0.6352     | 0.5182     |
| -----------                         | ---------          | ----------- | --------- | ------     | --------   |
|                                     | mouse\_0           | 0.7836      | 0.7232    | **0.7874** | 0.6823     |
|                                     | mouse\_1           | **0.9041**  | 0.8846    | 0.8658     | 0.8345     |
|                                     | mouse\_2           | **0.8811**  | 0.8598    | 0.8658     | 0.8226     |
|                                     | mouse\_3           | **0.8033**  | 0.6864    | 0.7679     | 0.7531     |
|                                     | mouse\_4           | **0.7029**  | 0.6684    | 0.6301     | 0.6265     |
| **TF Binding (Mouse)**              | *Average*          | *0.815*     | *0.7644*  | *0.7834*   | *0.7654*   |
| -----------                         | ---------          | ----------- | --------- | ------     | --------   |
|                                     | tf\_0              | **0.8251**  | 0.8246    | 0.8234     | 0.7989     |
|                                     | tf\_1              | **0.8553**  | 0.8438    | 0.8418     | 0.8316     |
|                                     | tf\_2              | **0.814**   | 0.7798    | 0.7573     | 0.7538     |
|                                     | tf\_3              | **0.7518**  | 0.7367    | 0.6812     | 0.6915     |
|                                     | tf\_4              | **0.8571**  | 0.8266    | 0.7906     | 0.8466     |
| **TF Binding (Human)**              | *Average*          | *0.8207*    | *0.8023*  | *0.7789*   | *0.7654*   |
| -----------                         | ---------          | ----------- | --------- | ------     | --------   |
|                                     | prom\_300\_all     | **0.925**   | 0.9112    | 0.9148     | 0.8804     |
|                                     | prom\_300\_notata  | 0.9604      | 0.9514    | **0.9646** | 0.9203     |
|                                     | prom\_300\_tata    | 0.8051      | 0.8077    | **0.8091** | 0.6719     |
|                                     | prom\_core\_all    | **0.8272**  | 0.8139    | 0.7863     | 0.762      |
|                                     | prom\_core\_notata | **0.8349**  | 0.8312    | 0.8283     | 0.7849     |
|                                     | prom\_core\_tata   | **0.8789**  | 0.8691    | 0.7341     | 0.721      |
| **Promoter Detection (Human)**      | *Average*          | *0.8794*    | *0.8641*  | *0.8395*   | *0.7901*   |
| -----------                         | ---------          | ----------- | --------- | ------     | --------   |
|                                     | EMP\_H3K14ac       | 0.7482      | 0.7539    | 0.7944     | **0.82**   |
|                                     | EMP\_H3K36me3      | 0.7555      | 0.7648    | 0.773      | **0.8027** |
|                                     | EMP\_H3K4me1       | 0.7319      | 0.7217    | 0.7291     | **0.7709** |
|                                     | EMP\_H3K4me2       | 0.7373      | 0.7381    | 0.7328     | **0.7777** |
|                                     | EMP\_H3K4me3       | 0.6746      | 0.6710    | 0.6595     | **0.7775** |
|                                     | EMP\_H3K79me3      | **0.8091**  | 0.8061    | 0.8012     | **0.8085** |
|                                     | EMP\_H3K9ac        | 0.7784      | 0.7524    | 0.7321     | **0.7846** |
|                                     | EMP\_H4            | **0.8828**  | 0.8826    | 0.8746     | 0.8199     |
|                                     | EMP\_H4ac          | 0.7108      | 0.7150    | 0.6911     | **0.782**  |
|                                     | EMP\_H3            | 0.8575      | 0.8483    | **0.8749** | 0.8242     |
| **Histone Mark Prediction (Yeast)** | *Average*          | *0.7639*    | *0.7653*  | *0.7548*   | *0.7968*   |

2. token-level tasks
# Macro F1-score on Token-level Annotation Tasks

All baselines adopt the conventional fine-tuning setup with a frozen pretrained encoder and a task-specific CNN-based decoder head. **NT-500M-FT**, **EVO2-FT**, and **CD-GPT-FT** refer to decoder-head fine-tuned variants of their respective base models.

| Model              | Benchmark Task (CDXBench) | Zero-shot Annotation |        |        |        |
| ------------------ | ------------------------- | -------------------- | ------ | ------ | ------ |
|                    |                           | CDS                  | EXON   | mRNA   | lncRNA |
| NT-500M-FT         | 0.3430                    | -                    | -      | -      | -      |
| EVO2-FT            | 0.2820                    | -                    | -      | -      | -      |
| CD-GPT-FT          | 0.3182                    | -                    | -      | -      | -      |
| Geno-Base-FT       | 0.3916                    | -                    | -      | -      | -      |
| Geno-Base-Continua | **0.4292**                | -                    | -      | -      | -      |
|                    |                           |                      |        |        |        |
| Geno-Zero          | 0.5472                    | 0.6562               | 0.7546 | 0.5753 | 0.5813 |
