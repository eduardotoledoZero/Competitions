# Competition 1: Natural Processing Language/ Movies Genres Classification

This project will use a set of movies genres data . Each observation has the titile of the movie, year of the movie, title plot and the different genres of the movie (a movie might be classified to multiples genres).

Title: 'How to Be a Serial Killer'</br>
Plot: 'A serial killer decides to teach the secrets of his satisfying career to a video store clerk.'</br>
Genres: 'Comedy', 'Crime', 'Horror'

![image info](https://eduardotoledozero.github.io/assets/img/competitions/moviegenre.png)

The idea is to use this data in order to predict the likely of a movie belongs to a genre given its plot.

Thank to Professor Fabio González, Ph.D. and his student John Arevalo by providing  this dataset.(UniAndes) See <https://arxiv.org/abs/1702.01992>

In this project, the full Machine Learning pipeline can be seen in each part of this work:

- Exploratory  Data Analysis
- Data Cleansing in order to remove noisy
- Text Preprocessing in order to reduce the corpus of the vocabulary.
- Text tokenization/vectorization
- Modeling  
- Model Hyperparameter tuning
- Prediction
- Model Deployment to production  

## 1. Exploratory  Data Analysis

### 1.1. The most frecuently used genres

This visualization shows this problem is characterized by a high presence of imbalanced classes (genrs) in which drama is most frecuently used genre and news is the least used  </br>
![The most frequently used genres](https://eduardotoledozero.github.io/assets/img/competitions/nlp/movies_genres_classification/mostusedgenres.jpg)

Also, there are 2405 movies classified with two genres, 2419 classified with 3 genres, 1172 classified with one genre and 2 movies with 9 genres such as the the below visualiztion shows:
![Movies Quantity by number of genres](https://eduardotoledozero.github.io/assets/img/competitions/nlp/movies_genres_classification/frequencyofgenres.jpg)

### 1.2. Frequency of use of words

The below visualization shows the 100 words most used without any type of preprocessing.

![Frequency of use of words](https://eduardotoledozero.github.io/assets/img/competitions/nlp/movies_genres_classification/fequencyofuseofwords.jpg)

At this point, the length of the vocabulary of full corpus is  39646 words

This notebook can be checked in  <https://github.com/eduardotoledoZero/Competitions/blob/main/NLP/Movies_Genres_Classification_EDA.ipynb>

## 2. Text Preprocessing

It is the first step in the pipeline  of **Natural Language Processing**  in order to clean the text, reduce the vocabulary to get the words with greater predictive value and extract meaningful information and hence, prepare it for the phase of model building.
The preprocessing steps are:

- Lower Casing
- Removal of Punctuation Mark
- Removal of Stop Word 
- Stemming or Lemmatization
- Tokenization

In order to apply these steps, NLTK -Natural Language Toolkit - is used. NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries

### 2.1. Lower Casing

In any sentence, you can find words in lower case and upper case at same time. It creates an issue because the algorithms of NLP see it as different entities of information. Therefore, the first transformation is to convert to lower casing and achieve uniformity in the corpus of the vocabulary.

### 2.2. Removal of Punctuation Mark 

Here, the punctuations marks  and html tags  are removed due to the business case which is to predict the genre. However, if the business case was related to emotions classification/ sentimental analysis then the text preprocessing should keep/retain some punctuation marks.

In this competition, the steps mentioned above were applied in a row and the below visualization shows the reduction of vocabulary and the change in the composition of the corpus.

![Frequency of use of words](https://eduardotoledozero.github.io/assets/img/competitions/nlp/movies_genres_classification/textprocessing_1.jpg)

After these transformations , the length of the vocabulary of full corpus is  39428 words.

### 2.3. Removal of Stop words

A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore.</br>
Given that Stop words  do not add much value to global meaning  of the sentence, then they can be removed and it would help a lot to reduce the vocabulary and thus, it helps to the reduction of dimensionality.
NLTK(Natural Language Toolkit) in python has a list of stopwords stored in 16 different languages.

![Frequency of use of words](https://eduardotoledozero.github.io/assets/img/competitions/nlp/movies_genres_classification/stopword_removal.jpg)

After this transformation , the length of the vocabulary of full corpus is  39373 words.

### 2.4. Stemming/Lemmatization

Stemming is the process of reduction of a word into its root word by removing derivational suffixes. For example, the words “likes”, “likely” and “liked” all result in the common root “like”. Sometimes, the result of stemming has no a meaning. For example, for universe, universal, university; their root is univers.

Lemmatization: uses context  to transform words to their dictionary(base) form. In general term, lemmatization is preferred to stemming because it adds context and the context is important in NLP applications.

The below visualization reflects the change of composition of the vocabulary after lemmatization.

![Frequency of use of words](https://eduardotoledozero.github.io/assets/img/competitions/nlp/movies_genres_classification/lemmatization.jpg)

Hence, after lemmatization , the length of the new vocabulary is 34211

### 2.4. Tokenization

In a large text corpus, some words will be very present (e.g. “the”, “a”, “is” in English) hence carrying very little meaningful information about the actual contents of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms.

In this competition , **TfIdVectorizer** to convert a collection of raw documents to a matrix of TF-IDF features in which are based on the logic that words that are too abundant in a corpus and words that are too rare are both not statistically important for finding a pattern.

Once ***TfIdVectorizer*** is applied, the length of the vocabulary is 17356. It helps a lot to reduce the dimensionality.

This notebook can be checked in  <https://github.com/eduardotoledoZero/Competitions/blob/main/NLP/Movies_Genres_Classification_TextPreprocessing.ipynb>

## 3. Model Building

Here will be presented three approaches to build the model through a classical machine learning model as RandomForest , a machine learning with deep neural network and Long short term-memory networks respectively.

### 3.1. Classic Machine Learning Model: RandomForest

Given the nature  of this bussines problem in where a movie can belong to  multiple genres ,  it is categorized as a multi-label classification in which each label is  not mutuallly exclusive and they are somehow related.</br>

First of all, the genres -target variable- has to be encoded appropiately. In order to do it, sklearn offers **MultiLabelBinarizer**. Then, split  data into training set and test set. 

Consecutively, Hyperparameter tuning is carried out and essentially , this process gets the best set of parameters selected for the model. There are many hypeparameters optimization frameworks as Optuna, HyperOpt among others but also, the existing two generic approaches such as GridSearchCV and RandomizedSearchCV which come into sklearn. However,  I recently found a new one with dask called Dask-SearchCV which is ***distribuited hyperparameter optimization with Scikit-Learn*** being this version with Dask faster that  the sckit-learn version and gave one try to it to see its behavior.

GridSearchCV is basically considering all the combinations of the candidates in finding the best parameters. This would in turn take a very long time when there are a greater number of parameter and their values to tune. Instead, RandomizedSearchCV solves the drawbacks of GridSearchCV, as it goes through only a fixed number of hyperparameter settings. It moves within the grid in random fashion to find the best set hyperparameters. This approach reduces unnecessary computation.

Here, RandomizedSearchCV is used to get a baselines of parameter and reduce the scope of the grid in order to be used in Dask-SearchCV.
The best score as result of the hyperparameters tuning is a AUC=0.85 and its best parameters are 

{'estimator__n_estimators': 1000, 'estimator__min_samples_split': 20, 'estimator__max_leaf_nodes': 30, 'estimator__max_features': 'sqrt'}

Finally, given these parameters, the model  with the dataset testing is ran and its AUC is 0.8585677305266167. It means that the model generalizes well in both datasets.


This notebook can be checked in  <https://github.com/eduardotoledoZero/Competitions/blob/main/NLP/Movies_Genres Classification_Modelling_RandomForest.ipynb>
