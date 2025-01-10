# Mood Classification of Songs Based on Lyrics

This project uses natural language processing (NLP) and machine learning techniques to classify songs into different mood categories based on their lyrics. The Jupyter Notebook contains all steps from data preprocessing to model evaluation, making it easy to follow and replicate.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Notebook Structure](#notebook-structure)
- [Usage](#usage)
- [Results](#results)
- [Future Scope](#future-scope)
- [References](#references)

## Introduction
Song lyrics often reflect the mood or sentiment of a piece. This project analyzes song lyrics to classify them into predefined mood categories such as happy, sad, energetic, or calm. It leverages advanced NLP techniques and machine learning algorithms to achieve accurate predictions.

## Features
- **Text Preprocessing**: Includes tokenization, stop-word removal, stemming, and lemmatization.
- **Feature Extraction**: Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) and word embeddings.
- **Machine Learning Models**: Implements classifiers like Logistic Regression, Support Vector Machines (SVM), and Neural Networks.
- **Interactive Visualizations**: Confusion matrices and performance metrics are plotted for better understanding.

## Dataset
- The dataset includes song lyrics paired with mood labels.
- Preprocessed to remove noise and ensure compatibility with NLP techniques.
- Source: Publicly available lyric datasets or user-curated collections.

## Technologies Used
- **Languages**: Python (Jupyter Notebook)
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, NLTK, TensorFlow/Keras
- **Tools**: Jupyter Notebook for interactive data exploration and analysis.

## Notebook Structure
1. **Data Preprocessing**:
   - Cleans and prepares the dataset for feature extraction.
2. **Feature Extraction**:
   - Uses TF-IDF for traditional machine learning models.
   - Implements word embeddings (e.g., Word2Vec, GloVe) for deep learning models.
3. **Model Training**:
   - Trains Logistic Regression, SVM, and Neural Network models on the dataset.
4. **Evaluation**:
   - Calculates accuracy, precision, recall, and F1-score.
   - Visualizes results using confusion matrices.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/rajvi-patel-22/Mood-classification-of-songs-based-on-lyrics.git
2. Run the cells sequentially to preprocess data, train models, and evaluate results.
## Results
- Achieved high accuracy in mood classification using SVM with TF-IDF.
- Neural networks demonstrated strong performance when using word embeddings for feature extraction.
- Visualizations of confusion matrices provide detailed insights into model performance.
<div style="background-color: white; padding: 10px; display: inline-block;">
  <img src="https://github.com/user-attachments/assets/ab5604d8-4d02-4e7e-9b1c-f4be11a2ef60" alt="download">
</div>


## Future Scope
- Extend the dataset to include multilingual song lyrics.
- Integrate transformer-based models like BERT for improved contextual analysis.
- Develop a web-based interface for real-time mood classification.
## References
- Natural Language Toolkit (NLTK) Documentation
- Scikit-learn User Guide
- TensorFlow/Keras Documentation
- Research papers on sentiment analysis and mood classification.
