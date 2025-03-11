# ADA_SentimentAnalysis
Notebook for the course Advanced Data Analytics

The goal of this project is to compare two different models and their performance when tasked with labeling a movie review as positive or negative. Both models where trained using the imdb review dataset.

**The dataset**: The imdb reviews is a balanced dataset which includes a total of 50.000 reviews labeled as positive or negative.

### Models used
1. TF-IDF with Naive-Bayes [(In detail)](#model-1).
2. Word2Vec with LSTM

Steps taken to preprocess text data:
1. For Models 1 and 2
* Removing HTML tags using BeautifulSoup
* Remove punctuation
* Convert to lowercase
* Remove double whitespaces
* Remove singular characters(characters surrounded by whitespace)
2. Only for Model 2
* Tokenization
* Remove stopwords(Using the "nltk stopwords" list)

## Model 1
- TF-IDF is used to vectorize the vocabulary(all unique words from the training data) and give them a score based on their importance in a review. The score of a term increases when its rare across all documents or when its important for the document(found multiple times). This process will be applied to each term in the vocabulary and the result will be a sparse(mostly 0's) TF-IDF matrix:
- Naive-Bayes is used 
## Model 2
