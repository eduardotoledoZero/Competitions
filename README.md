# Competitions

## Objective

This repository containls all muy work for all competitions in which I have participartedin Kaggle and Zindi.Africa.

## Competition 1: Natural Processing Language

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

### Exploratory  Data Analysis

- The most frecuently used genres

This visualization shows this problem is characterized by a high presence of imbalanced classes (genrs) in which drama is most frecuently used genre and news is the least used![image info](https://eduardotoledozero.github.io/assets/img/competitions/mostusedgenres.jpg)


This notebook can be checked in  <https://github.com/eduardotoledoZero/Competitions/blob/main/Movies_Genres_Classification_EDA.ipynb>


