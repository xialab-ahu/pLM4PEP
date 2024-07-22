# pLM4PEP

plant miRNA encoded peptides prediction based on protein language model

## Introduction

Movitation:

Plant miPEPs play a crucial role in regulating diverse plant traits. Plant miPEPs identification is challenging due to limitations in the available number of known miPEPs for training. Meanwhile, existing prediction methods rely on manually encoded features to infer plant miPEPs. Recent advances in deep learning modeling of protein sequences provide an opportunity to improve the representation of key features, leveraging large datasets of protein sequences. In this study, we develop a prediction model based on ESM2 to achieve accurate identification of plant miPEPs.

Results:

We propose a prediction model, named pLM4PEP, to predict plant miPEPs. In this field, compared with state-of-the-artl predictor,  pLM4PEP enables more accurate prediction and stronger generalization ability. pLM4PEP utilizes ESM2 to extract peptide feature embeddings, with LR serving as the classifier. The validation experiments conducted on various biopeptide datasets show that pLM4PEP has superior prediction performance.

<img src="file:///C:/Users/yys/AppData/Roaming/marktext/images/2024-05-15-11-18-45-image.png" title="" alt="" width="637">

## Related Files

| FILE NAME               | DESCRIPTION                                                                                   |
|:----------------------- |:--------------------------------------------------------------------------------------------- |
| main.py                 | the main file of pLM4PEP predictor (include data reading, encoding, and classifier selection) |
| ML_grid_search_model.py | Grid search method for optimizing the structure of neural network                             |
| datasets                | data                                                                                          |
| esm2                    | code for extracting the embedding layer                                                       |
| pydpi                   | protein calculation package                                                                   |
| feature                 | protein characterization calculation library                                                  |

## Installation

- Requirement
  
  OS：
  
  - `Windows` ：Windows10 or later
  
  - `Linux`：Ubuntu 16.04 LTS or later

Python：

- `Python` >= 3.8
  
  - Download `pLM4PEP`to your computer
  
  ```bash
  git clone https://github.com/xialab-ahu/pLM4PEP.git
  ```

- open the dir in `conda prompt` and ceate a new environment named `myenv` with `environment.yaml`
  
  ```
  cd pLM4PEP
  conda env create -f environment.yaml -n myenv
  ```

## Contact

Please feel free to contact us if you need any help.
