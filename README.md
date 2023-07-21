# Course Review Sentiment Analysis: A Comparative Study Of Machine Learning and Deep Learning Methods

In this ever-evolving world of education, one of the key tools for improving the quality of teaching and learning is understanding student sentiment and feedback through course evaluations. This study analyzed more than 100,000 review of courses on the online learning platform Coursera, with the objective of comparing the efficacy of various sentiment analysis methods. The initial preprocessing steps involved data cleaning using different Natural Language Processing (NLP) techniques. The cleaned data subsequently underwent transformation into TF-IDF and word embeddings were created using a pre-trained Twitter-GloVe model. Naive Bayes, Random Forest, and SVM models were trained on the TF-IDF vectorized data, and fine-tuned using a grid search method. Meanwhile, the word embeddings were used to train the LSTM and GRU models, which were optimized through Bayesian methods. Additionally, the BERT model was incorporated for further comparison.  The findings revealed that BERT outperformed all other models in metrics such as accuracy, precision, recall, and F1 score, making it the most effective for the sentiment analysis of course evaluations in this study. This comprehensive analysis aims to offer valuable insight into the sentiments of course evaluation, ultimately striving to improve the educational experiences of students. It also discusses the significance of the findings and highlights potential areas of improvement for future research.


## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#ackonwledgments)


## Getting Started
Follow this simple guide to get started with setting up the project locally and running it.

### Prequisites
To run the code you'll need:

- Python 3.10.8 or higher
- Course reviews dataset available at the link: [course reviews](https://www.kaggle.com/datasets/septa97/100k-courseras-course-reviews-dataset).

### Installation
1. Clone the repository:
'''bash
git clone https://github.com/ekinfergan/Thesis_Jupyter_Final.git

2. Install the required packages using the following command:

pip install -r dependencies.txt

## Usage
This project follows a specific pipeline from preprocessing the data to training the models and generating the results. Follow the steps below to run the entire project:

1. **Preprocessing**
    - Start by running `cleaning.ipynb` in the `preprocessing` directory. This notebook cleans the review (text) part of the original data and removes unnecessary noise using various Natural Language Processing techniques. 
    - Run `processor.ipynb` and `processor_neg.ipynb`, both found in the `preprocessing` directory. These notebooks prepare the data for the upcoming steps by performing further cleaning and preprocessing. The new datasets that are created will be saved in the `input/processed/normal` and `input/processed/neg-tagged` directories.

2. **Vectorization**
    - Run `tfidf_generator.ipynb` in the `preprocessing` directory. This notebook will create TF-IDF vectors from the `neg-tagged` data and save the resulting vectors into the `input` directory under `neg-tagged`.
    - Run `embeddings_generator.ipynb` in the `preprocessing` directory. This notebook will generate word embeddings from the `normal` data and save the resulting vectors into the `input` directory under `normal`.

3. **Modeling**
    - In the `models` directory, run `classical_ml.ipynb` to apply Naive Bayes, Random Forest and SVM on the TF-IDF data.
    - Then run `rnns.ipynb` to apply LSTM and GRU on the word embeddings data. 
    - Lastly, run `bert.ipynb` to apply the BERT model. <br>
    Note: The models can also be run in Google Collab.

4. **Results**
    - The results from the classical machine learning models and RNNs are saved into the `results` directory. Each model will have its own subdirectory inside the `results` directory where its specific results will be saved.

Please note that you should have the necessary dependencies installed, and you should replace any placeholders in the code with your actual data.


## License
This project is a part of the Bachelor Thesis at the University of Groningen and is for academic use only. 


## Contact
- Ekin Fergan - e.fergan@student.rug.nl
- Project Link: [https://github.com/ekinfergan/Thesis_Jupyter_Final.git](https://github.com/ekinfergan/Thesis_Jupyter_Final.git) 


## Acknowledgments
I would like to express my gratitude towards my supervisor Dr. Tsegaye Misikir Tashu and the University of Groningen for their continual support throughout this project. 

References:
- [Word Frequency Histogram](https://www.kaggle.com/code/pamin2222/tf-idf-svm-exploration)
- [Undersampling and Oversampling Strategies](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/)
- [Machine Learning Mastery - Word Embeddings](https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/)
- [Sentiment Analysis with TF-IDF and Random Forest](https://www.kaggle.com/code/onadegibert/sentiment-analysis-with-tfidf-and-random-forest/notebook)
- [Gradient Boost Model](https://medium.com/@crawftv/parameter-hyperparameter-tuning-with-bayesian-optimization-7acf42d348e1)
- [Word2Vec - LSTM](https://www.kaggle.com/code/caiyutiansg/twitter-sentiment-analysis-with-word2vec-lstm)
- [Fine-tuning BERT](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=mjJGEXShp7te )
- [Sentiment Analysis - BERT](https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert#Sentiment-Analysis-using-BERT)