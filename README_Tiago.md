# Automated Customer Reviews Sentiment Analysis

This project applies sentiment analysis to Amazon product reviews using two distinct approaches:  
1. **DistilBERT-based NLP model** for deep learning.
2. **Traditional machine learning model** using FastText embeddings.  

The aim is to compare these methods for classifying reviews as negative, neutral, or positive.

## Project Overview
This project was undertaken as part of the Data Science & Machine Learning bootcamp at Ironhack. It showcases the application of Natural Language Processing (NLP) techniques, comparing state-of-the-art deep learning models with more traditional machine learning methods.

## File Structure
├── data/
│   ├── train.csv                # Training data (Feb-Apr 2019)
│   ├── test.csv                 # Test data (Sep-Oct 2018)
│
├── models/
│   ├── distilbert_model/        # Directory to save the DistilBERT model
│   ├── traditional_model.pkl    # Pickle file for the best traditional model
│
├── notebooks/
│   ├── distilbert_analysis.ipynb # Jupyter notebook for the DistilBERT approach
│   ├── traditional_ml.ipynb      # Jupyter notebook for the traditional ML approach
│
├── scripts/
│   ├── preprocess.py            # Script for data preprocessing
│   ├── train_distilbert.py      # Script to fine-tune DistilBERT
│   ├── train_ml_model.py        # Script for traditional ML model training and evaluation
│
├── results/
│   ├── distilbert_results.txt   # Evaluation results for DistilBERT
│   ├── traditional_results.txt  # Evaluation results for traditional ML approach
│
├── requirements.txt             # List of required libraries
├── README.md                    # Project documentation
└── LICENSE                      # Project license

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Amazon_SentimentAnalysis

2. **Create a virtual environment (recommended)**:
    python -m venv venv
    source venv/bin/activate   # On Linux/macOS
    venv\Scripts\activate      # On Windows

3. **Install dependencies**:
pip install -r requirements.txt

## How It Works

1. **Data Preprocessing**:
   - Run `scripts/preprocess.py` to clean and prepare the data.
   - **Text Cleaning**: Removes special characters, converts text to lowercase, and eliminates stopwords.
   - **Vectorization**: Uses FastText embeddings for traditional ML models.

2. **DistilBERT-based NLP Model**:
   - Run `scripts/train_distilbert.py` to fine-tune the DistilBERT model.
   - **Preprocessing**: Tokenizes text for the DistilBERT model.
   - **Training**: Fine-tunes using a pre-trained DistilBERT model from Hugging Face.
   - **Evaluation**: Computes metrics like accuracy, precision, recall, and F1-score.

3. **Traditional Machine Learning Model**:
   - Run `scripts/train_ml_model.py` to train and evaluate traditional ML models.
   - **Vectorization**: Uses FastText embeddings for text vectorization.
   - **Model Training**: Implements Logistic Regression, SVM, and Random Forest.
   - **Model Selection**: Uses cross-validation and grid search for tuning.

4. **View Results**:
   - Check the `results` folder for performance metrics and comparisons.


## Performance Evaluation

1. **DistilBERT Model**:
   - Evaluated using metrics such as:
     - **Accuracy**: Measures the overall correctness of predictions.
     - **Precision**: Indicates the accuracy of positive predictions.
     - **Recall**: Measures the ability to capture all positive instances.
     - **F1-Score**: Harmonic mean of precision and recall for balanced evaluation.
   - **Confusion Matrix**: Provides detailed insights into misclassifications and helps understand model performance.

2. **Traditional Machine Learning Models**:
   - **Cross-Validation**: Used to compare multiple models and select the best-performing one.
   - **Grid Search**: Performed to fine-tune hyperparameters of the selected model.
   - **Evaluation Metrics**: Accuracy, precision, recall, and F1-score are used to assess performance.

## Example Results

1. **DistilBERT**: Achieved higher accuracy due to its ability to understand deep contextual nuances in text.
2. **Traditional Models**: Showed good performance with FastText embeddings but struggled with comprehending deeper semantic contexts compared to DistilBERT.

## Requirements

- **Python 3.8+**
- **Dependencies**: Listed in `requirements.txt`. To install, run:
  ```bash
  pip install -r requirements.txt

## Future Work
1. **Enhanced Preprocessing**: Experiment with more sophisticated text cleaning techniques to improve model performance.
2. **More Models**: Explore models like BERT, RoBERTa, and GPT for performance comparison.
3. **Larger Datasets**: Use a broader range of datasets for better generalization and robustness.
4. **Deployment**: Develop a user-friendly interface for real-time sentiment analysis.

## Acknowledgments
This project was completed as part of the Data Science & Machine Learning Bootcamp at Ironhack. Special thanks to:
- Instructors and Mentors: For their continuous guidance and support.
- Peers: For collaboration, discussions, and shared learning experiences.


