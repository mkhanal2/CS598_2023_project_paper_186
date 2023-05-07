# Improving clinical outcome predictions using convolution over medical entities with multi modal learning  - Reproducing paper for CS598 DL4H in Spring 2023
Project to replicate the results for paper Improving clinical outcome predictions using convolution over medical entities with multi modal learning

- Link to the Original Paper [[1]](#1): https://pubmed.ncbi.nlm.nih.gov/34127241/ 
- Link to the code for original paper [[2]](#2): https://github.com/tanlab/ConvolutionMedicalNer

## Code Dependencies
 Python 3 was used for this project. Here are the python modules used for the project.
 - pandas, os, numpy, torch, spacy, gensim.models (Word2Vec), sklearn

For Word2Vec [[6]](#6) pre-trained model was used. Here is the link to download the Word2Vec model.
- Download Pre-trained Word2Vec [[7]](#7) : https://github.com/kexinhuang12345/clinicalBERT

For Med7 [[5]](#5) entity extraction, here is the package that needs to be intalled.
- pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl

## Data Download Instructions 
We need two different data files for this project. You would need to have access to MIMIC-III dataset through PhysioNet [[3]](#3). Here are the details on how to download them:

- **all_hourly_data.h5** - You can use pre-processed data by the MIMIC-III extract project [[4]](#4). PhysioNet provides linking the google account to make sure if you download access through google accoutn. Here is the direct link to google storage to get this data.  https://console.cloud.google.com/storage/browser/mimic_extract
- **NOTEEVENTS.csv.gz** - You can downlaod this one file directly from the PhysioNet MIMIC-III page. Here is the link to that main page which has data in the end section once you login. https://physionet.org/content/mimiciii/1.4/

## Code Run Instructions

After installing all the required pacakges above and downloading the required data and pre-trained modles, here are the setps that can be followed to run this code:

1. There are three code files for this project, two library files (*project_data_prep_lib.py* and *project_model_lib.py*) and the main code is part of the notebook *Project Main.ipynb*. Keep all of these files in main folder. 
2. Create a data folder under the main folder. Move data files (all_hourly_data.h5 and NOTEEVENTS.csv.gz) into the data folder. (set variables MIMIC_EXTRACT_FILE and CLINICAL_NOTES_FILE accrodingly with is path). 
3. Also, move pre-trained Word2Vec model (word2vec.model) into the data folder. (set var WORD_2_VEC_MODEL_PATH within project_data_prep_lib.py with this.) 
4. Create a temp folder under the main folder which will be used for storing files in temporary basis. (set var TEMP_DATA_DIR within main notebook with this.) 
5. Create a saved_model folder under the main folder which will be used for storing the final trained model. (set var MODEL_SAVE_PATH within project_model_lib.py with this.)
6. Once we have all the above setup we should be able to run the overall code. The notebook is sub-divided into different section for data-prep, visualizatoin, feature-creation, Model training, testing and validation, and finally there is also a discussion section. 
7. Run all of the cells in sequential order. 

## References
<a id="1">[1]</a> 
Batuhan Bardak and Mehmet Tan (2021). 
Improving clinical outcome predictions using convolution over medical entities with multimodal learning. Artificial Intelligence in Medicine, Volume 117,
https://doi.org/10.1016/j.artmed.2021.102112

<a id="2">[2]</a> 
Batuhan Bardak and Mehmet Tan (2021).
Code for paper - Improving clinical outcome predictions using convolution over medical entities with multimodal learning. https://github.com/tanlab/ConvolutionMedicalNer

<a id="3">[3]</a> 
Johnson, A., Pollard, T., and Mark, R. (2016). 
MIMIC-III Clinical Database (version 1.4). PhysioNet. https://doi.org/10.13026/C2XW26

<a id="4">[4]</a> 
Shirly Wang, Matthew B. A. McDermott, Geeticka Chauhan, Michael C. Hughes, Tristan Naumann, 
and Marzyeh Ghassemi. 
MIMIC-Extract: A Data Extraction, Preprocessing, and Representation 
Pipeline for MIMIC-III. arXiv:1907.08322. https://github.com/MLforHealth/MIMIC_Extract

<a id="5">[5]</a> 
Kormilitzin, Andrey and Vaci, Nemanja and Liu, Qiang and Nevado-Holgado, Alejo (2020). 
Med7: a transferable clinical natural language processing model for electronic health records. arXiv preprint arXiv:2003.01271. https://github.com/kormilitzin/med7

<a id="6">[6]</a> 
Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean. 
Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781v3. https://doi.org/10.48550/arXiv.1301.3781

<a id="7">[7]</a> 
Kexin Huang, Jaan Altosaar, Rajesh Ranganath 
This repo hosts pretraining and finetuning weights and relevant scripts for ClinicalBERT. https://github.com/kexinhuang12345/clinicalBERT
