# Fake News Prediction
This project implements a machine learning model to distinguish between real and fake news articles using logistic regression. By preprocessing the text data and converting it into a numerical format, the model identifies patterns and features that help determine the authenticity of the content.
## Table of Contents
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Dependencies](#dependencies)  
- [Data Pre-processing](#data-pre-processing)  
- [Model Training](#model-training)  
- [Evaluation](#evaluation)  
- [Usage](#usage)  
- [Results](#results)  
## Project Overview  
This project leverages natural language processing (NLP) and machine learning techniques to identify fake news. The key steps include data cleaning, text preprocessing (such as stemming and stopword removal), converting text into numerical representations, and training a logistic regression model on the processed data.  

## Dataset  
The dataset used in this project, `data.csv`, consists of labeled news articles. It includes columns for the author, title, and a label indicating whether the news is fake (1) or real (0).  
## Dependencies  
Ensure you have the following Python libraries installed:  

- `numpy`  
- `pandas`  
- `nltk`  
- `sklearn`  

You can install these packages using the following command:  

```sh
pip install numpy pandas nltk scikit-learn
```
## Data Pre-processing  
- **Loading the Data**: Load the dataset and replace any missing values with empty strings.  
- **Text Processing**: Merge the `author` and `title` fields into a single `content` field.  
- **Stop Words Removal**: Eliminate common words that do not significantly contribute to model predictions, such as "the" and "is".  
## Model Training  
The model used for this project is **Logistic Regression**, a classification algorithm that performs well for text classification tasks.  

- **Text Vectorization**: Use `TfidfVectorizer` to convert text data into numerical vectors.  
- **Splitting the Data**: Divide the dataset into training and test sets (80/20 split) using `train_test_split` from `sklearn`.  
- **Training**: Train a logistic regression model on the training data.  

## Evaluation  
The model's accuracy is evaluated using the accuracy score metric:  

- **Training Accuracy**: Measures how well the model fits the training data.  
- **Test Accuracy**: Assesses the model's performance on unseen data.  

## Usage  
To use the model for predictions:  

1. Run the code to load and preprocess the dataset.  
2. Train the model.  
3. Test the model with a sample input to classify the news as real or fake.  
## Results  
The final model provides accuracy scores for both the training and test datasets, helping to evaluate its performance.  

## Acknowledgments  
Thanks to the dataset provider and the NLP community for providing tools and resources that enable the development of such models.  
