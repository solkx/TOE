# [TOE: A Grid-Tagging Discontinuous NER Model Enhanced by Embedding Tag/Word Relations and More Fine-Grained Tags](https://ieeexplore.ieee.org/document/9944897)

## 1. Environments
```text
- python (3.8)
- cuda (11.4)
- torch (1.8.1)
- pip install -r requirements.txt
```

## 2. Dataset
   * [CADEC](https://pubmed.ncbi.nlm.nih.gov/25817970/)  
   * [ShARe13](https://clefehealth.imag.fr/?page_id=441)  
   * [ShARe14](https://sites.google.com/site/clefehealth2014/)

## 3. Pre-training
Download the pre-training model at the following link:
   * BioBERT: [https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/tree/main](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/tree/main)  
   * ClinicalBERT: [https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/tree/main](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/tree/main)

## 4. Preparation
   * Get dataset 
   * Process them to fit the same format as the example in `data/`
   * Put the processed data into the directory `data/`

## 5. Training
```text
python main.py --config ./config/cadec.json
```