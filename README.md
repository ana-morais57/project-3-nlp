![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Fake News Detection Project

## Natural Language Processing Challenge

## Table of Contents
1. [Project Overview](#project-overview)
2. [Implementation Details](#implementation-details)
3. [Model Evaluation](#model-evaluation)
4. [Key Insights and Challenges](#key-insights-and-challenges)
5. [Usage Guide](#usage-guide)
6. [Future Work](#future-work)
7. [Files in Repository](#files-in-repository)

## Project Overview
### Objective
- Build a classifier to distinguish between real and fake news headlines

### Dataset
- **Source**: `dataset/training_data.csv`
    - Contains news headlines and labels:
        - **0**: Fake News
        - **1**: Real News
- Balanced dataset with approximately equal instances of fake and real news.
  
---

## Implementation Details

### Base Model
1. **Preprocessing**:
   - Tokenized text using **TF-IDF Vectorizer** with:
       - `stop_words='english'`
       - `max_features=5000`
       - `ngram_range=(1, 2)`
       - `max_df=0.8`
       - `min_df=5`
2. **Model**: Random Forest Classifier with:
   - `n_estimators=200`
   - `criterion='entropy'`
   - `max_depth=10`
3. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score.
  
### Transformer-Based Model
1. **Preprocessing**:
   - Removed duplicates.
   - Lowercased text and removed special characters.
   - Lemmatized tokens using `WordNetLemmatizer`.
   - Tokenized with Hugging Face’s `AutoTokenizer`.
2. **Model**:
   - **DistilBERT**: A lightweight transformer model.
   - Fine-tuned for binary classification using Hugging Face’s `Trainer` class.
3. **Training Configuration**:
   - Learning Rate: `2e-5`
   - Batch Size: `16`
   - Epochs: `3`
   - Maximum Sequence Length: `128`

---

## Model Evaluation
### Base Model
- Accuracy: 89%
  
### Transformer-Based Model
- Accuracy: 97%
- Classification Report:
    - Precision, Recall, and F1-Score improved significantly.
- Confusion Matrix: Reduced false positives and false negatives compared to the base model.

--

## Key Insights and Challenges
- **Key Insight**: Preprocessing was critical to model performance. Using lemmatization and removing stop words helped improve results.
- **Challenge**: Managing dependency conflicts between Python packages required careful debugging and reinstallation.
- **Learning**: DistilBERT outperformed the base model significantly due to its deep contextual understanding of text.

---

## Usage Guide

### Jupyter Notebook
1. Open the notebook file `NLPminiproject.ipynb`.
2. Follow the steps to preprocess data, train the model, and evaluate results.

### Streamlit App
1. Run the app using:
   ```bash
   streamlit run app.py
2. Input a news headline in the text box.
3. The app will predict whether the headline is real or fake, along with confidence scores.

---

## Future Work
- Experiment with additional transformer models such as BERT or RoBERTa.
- Explore token importance using advanced methods like SHAP.
- Improve the Streamlit app.

---

## Files in Repository

| File                         | Description                                                           |
|------------------------------|-----------------------------------------------------------------------|
| `requirements.txt `          | List of Python dependencies required for the project.                 |
| `NLPminiproject.ipynb`       | Jupyter Notebook documenting the analysis and model training.         |
| `app.py`                     | Streamlit application  to detect Fake News.                           |
| `saved_model/`               | 	Folder containing the tokenizer and model files required for the app.|
| `fakenews.pdf`               | Presentation detailing the project details and conclusions.           |
 
