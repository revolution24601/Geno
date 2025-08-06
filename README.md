# Foundation Model for Joint Reasoning Over Genomic Sequence Syntax and Biological Semantics

This repo contains the code and models for "[Foundation Model for Joint Reasoning Over Genomic Sequence Syntax and Biological Semantics]()".

## Contents
1. [Model](#model)
2. [Usage](#usage)
3. [Results](#results)

## Model
**Geno** is a biologically grounded foundation model that unifies sequence generation, modality translation, and token-level annotation within a single causal modeling framework. A three-phase pretraining curriculum, coupled with domain-informed architectural design and centered on task-conditioned prompting and dual-label supervision, enables bridging of sequence syntax and biological semantics across multiple levels of resolution. This design allows the model to perform both macro-scale regulatory inference and nucleotide-resolution annotation without task-specific architecture changes.

<img width="2020" height="902" alt="image" src="https://github.com/user-attachments/assets/15b7a089-7e21-47fb-b4a4-bbf9427a2f84" />

## Usage

This project incorporates key components from the [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA/tree/master) and [mamba_ssm](https://github.com/state-spaces/mamba/releases) repositories. All usage of code and concepts from these projects complies with the license terms specified in the original repository. We sincerely appreciate the MoonshotAI team and Mamba-ssm team for open-sourcing this valuable resource.

We use Conda for virtual environment management. An `environment.yml` file is provided for quick reproduction of the development environment.

Please follow the steps below:

1. Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
2. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/revolution24601/Geno.git
   cd Geno
   ```
3. Create and activate the Conda environment using the provided environment.yml file:
   ```bash
   conda env create -f environment.yml
   conda activate geno
   pip install mamba-ssm==2.2.2  # If there are issues during installation, you can manually download the file mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl and install it yourself.
   cd MoBA
   pip install .
   cd ..
   ```
4. Download checkpoints and tokenizer and put them under this directory
   ```bash
   mkdir ckpt
   ```
5. Run demo script:
   ```
   python demo.py
   ```
You can download the weights from:
- [Google Drive](https://drive.google.com/drive/folders/1WaoYB-azdB7oBOHiy7nzO-C4ObcB-H2u?usp=drive_link)
We have released the following checkpoints:

|          Checkpoint           | Description                                                                                                                                |
| :---------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           Geno-Zero           | checkpoint after Phase~III.                                                          |


## Results
1. sequence-level tasks
We evaluated our model on a diverse set of genomic tasks drawn from the GUE benchmark suites. These tasks are organized according to the core categories of biological inference, including the identification of the transcription factor (TF) binding site, the classification of the promoter and the core promoter, the recognition of the splice site, the prediction of histone marks and the detection of the long-range enhancer-promoter interaction.
Tasks are grouped by biological categories. Bolded values indicate the best score among the four models either per row (i.e., per dataset) or in the average row of each task type.

| Task Type                           | Dataset            | Geno-Zero-FT   | NT-500M-FT   | EVO2-FT       | CD-GPT-FT     |
| ----------------------------------- | ------------------ | ----------- | --------- | ---------- | ---------- |
| Splice Site                         | splice reconstruct | **0.8349**  | 0.6901    | 0.6352     | 0.5182     |
| -----------                         | ---------          | ----------- | --------- | ------     | --------   |
|                                     | mouse\_0           | 0.7836      | 0.7232    | **0.7874** | 0.6823     |
|                                     | mouse\_1           | **0.9041**  | 0.8846    | 0.8658     | 0.8345     |
|                                     | mouse\_2           | **0.8811**  | 0.8598    | 0.8658     | 0.8226     |
|                                     | mouse\_3           | **0.8033**  | 0.6864    | 0.7679     | 0.7531     |
|                                     | mouse\_4           | **0.7029**  | 0.6684    | 0.6301     | 0.6265     |
| **TF Binding (Mouse)**              | *Average*          | ***0.815***     | *0.7644*  | *0.7834*   | *0.7654*   |
| -----------                         | ---------          | ----------- | --------- | ------     | --------   |
|                                     | tf\_0              | **0.8251**  | 0.8246    | 0.8234     | 0.7989     |
|                                     | tf\_1              | **0.8553**  | 0.8438    | 0.8418     | 0.8316     |
|                                     | tf\_2              | **0.814**   | 0.7798    | 0.7573     | 0.7538     |
|                                     | tf\_3              | **0.7518**  | 0.7367    | 0.6812     | 0.6915     |
|                                     | tf\_4              | **0.8571**  | 0.8266    | 0.7906     | 0.8466     |
| **TF Binding (Human)**              | *Average*          | ***0.8207***  | *0.8023*  | *0.7789*   | *0.7654*   |
| -----------                         | ---------          | ----------- | --------- | ------     | --------   |
|                                     | prom\_300\_all     | **0.925**   | 0.9112    | 0.9148     | 0.8804     |
|                                     | prom\_300\_notata  | 0.9604      | 0.9514    | **0.9646** | 0.9203     |
|                                     | prom\_300\_tata    | 0.8051      | 0.8077    | **0.8091** | 0.6719     |
|                                     | prom\_core\_all    | **0.8272**  | 0.8139    | 0.7863     | 0.762      |
|                                     | prom\_core\_notata | **0.8349**  | 0.8312    | 0.8283     | 0.7849     |
|                                     | prom\_core\_tata   | **0.8789**  | 0.8691    | 0.7341     | 0.721      |
| **Promoter Detection (Human)**      | *Average*          | ***0.8794***    | *0.8641*  | *0.8395*   | *0.7901*   |
| -----------                         | ---------          | ----------- | --------- | ------     | --------   |
|                                     | EMP\_H3K14ac       | 0.7482      | 0.7539    | 0.7944     | **0.82**   |
|                                     | EMP\_H3K36me3      | 0.7555      | 0.7648    | 0.773      | **0.8027** |
|                                     | EMP\_H3K4me1       | 0.7319      | 0.7217    | 0.7291     | **0.7709** |
|                                     | EMP\_H3K4me2       | 0.7373      | 0.7381    | 0.7328     | **0.7777** |
|                                     | EMP\_H3K4me3       | 0.6746      | 0.6710    | 0.6595     | **0.7775** |
|                                     | EMP\_H3K79me3      | **0.8091**  | 0.8061    | 0.8012     | 0.8085 |
|                                     | EMP\_H3K9ac        | 0.7784      | 0.7524    | 0.7321     | **0.7846** |
|                                     | EMP\_H4            | **0.8828**  | 0.8826    | 0.8746     | 0.8199     |
|                                     | EMP\_H4ac          | 0.7108      | 0.7150    | 0.6911     | **0.782**  |
|                                     | EMP\_H3            | 0.8575      | 0.8483    | **0.8749** | 0.8242     |
| **Histone Mark Prediction (Yeast)** | *Average*          | *0.7639*    | *0.7653*  | *0.7548*   | ***0.7968***   |

2. token-level tasks
# Macro F1-score on Token-level Annotation Tasks

All baselines adopt the conventional fine-tuning setup with a frozen pretrained encoder and a task-specific CNN-based decoder head. **NT-500M-FT**, **EVO2-FT**, and **CD-GPT-FT** refer to decoder-head fine-tuned variants of their respective base models.
<table>
  <caption>
    Macro F1-score on token-level annotation tasks. All baselines adopt the conventional fine-tuning setup with a frozen pretrained encoder and a task-specific CNN-based decoder head. <strong>NT-500M-FT</strong>, <strong>EVO2-FT</strong>, and <strong>CD-GPT-FT</strong> refer to decoder-head fine-tuned variants of their respective base models.
  </caption>
  <thead>
    <tr>
      <th style="text-align: center; font-weight: bold; vertical-align: left;" rowspan="2">Model</th>
      <th style="font-weight: bold; vertical-align: middle;" rowspan="2">Benchmark Task (CDXBench)</th>
      <th colspan="4" style="font-weight: bold; text-align: center; vertical-align: middle;">Zero-shot Annotation</th>
    </tr>
    <tr>
      <th style="font-weight: bold; text-align: center;">CDS</th>
      <th style="font-weight: bold; text-align: center;">EXON</th>
      <th style="font-weight: bold; text-align: center;">mRNA</th>
      <th style="font-weight: bold; text-align: center;">lncRNA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">NT-500M-FT</td>
      <td>0.3430</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: center;">EVO2-FT</td>
      <td>0.2820</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: center;">CD-GPT-FT</td>
      <td>0.3182</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: center;">Geno-Base-FT</td>
      <td>0.3916</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: center;">Geno-Base-Continua</td>
      <td><strong>0.4292</strong></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: center; border-top: 2px solid #ddd;">Geno-Zero</td>
      <td style="border-top: 2px solid #ddd;">0.5472</td>
      <td style="border-top: 2px solid #ddd;">0.6562</td>
      <td style="border-top: 2px solid #ddd;">0.7546</td>
      <td style="border-top: 2px solid #ddd;">0.5753</td>
      <td style="border-top: 2px solid #ddd;">0.5813</td>
    </tr>
  </tbody>
</table>

