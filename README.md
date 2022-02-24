# TOE: A Grid-Tagging Model for Discontinuous Named Entity Recognition with Tag-Oriented Enhancement

---
### Model Framework

![avatar](figure/model_frame.png)

## 1. Environments
```text
- python (3.8.12)
- cuda (11.4)
```
## 2. Dependencies
```text
- numpy (1.21.4)
- torch (1.10.0)
- genism (4.1.2)
- transformers (4.13.0)
- pandas (1.3.4)
- scikit-learn (1.0.1)
- prettytable (2.4.0)
```
## 3. Dataset
   * [CADEC](https://pubmed.ncbi.nlm.nih.gov/25817970/)  
   * [ShARe13](https://clefehealth.imag.fr/?page_id=441)  
   * [ShARe14](https://sites.google.com/site/clefehealth2014/)

## 4. Preparation
   * Download dataset  
   * Process them to fit the same format as the example in `data/`
   * Put the processed data into the directory `data/`
## 5. Training
> python main.py --config ./config/cadec.json

## 6. License
   This project is licensed under the MIT License - see the LICENSE file for details.
