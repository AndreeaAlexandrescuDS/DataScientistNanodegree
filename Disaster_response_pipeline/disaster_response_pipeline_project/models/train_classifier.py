
# import libraries
import sys
import pandas as pd

from sqlalchemy import create_engine
import sqlalchemy as sql

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']) 

import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """
      Function that loads dataset from database and returns feature and target variables X and Y
    and the names of the targets (category_names).
      Arguments:
      --database_filepath: the path of the database (returned by process_data.py)      
    """
    
    # load data from database
    df = pd.read_sql_table('input_disaster', 'sqlite:///' + database_filepath)
    targets = [c for c in df.columns if c not in ['id', 'message', 'original', 'genre']]
    Y = df[targets]
    X = df['message']
    category_names = Y.columns
    return X,Y, category_names


def tokenize(text):
    """
      Function that removes english stop words from text,
    performs tokenization and lemmatization and returns 
    the tokens.   
      Arguments:
      --text: the text to be processed
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
      Function that builds a pipeline with:
      --vectorization: transformation of text in numerical features using Tfidf
      --classification: MultiOutputClassifier for multitarget classification problem
      --parameters tunning using Grid Search Cross Validation
      The function constructs the CV estimator.
    """
    
    # define pipeline
    pipeline = Pipeline([('vect', TfidfVectorizer(tokenizer=tokenize)), 
                         ('model', MultiOutputClassifier(RandomForestClassifier(), 
                                                         n_jobs=-1))])
    # define seach space for parameter tunning
    parameters = {
#             'vect__ngram_range': ((1, 1), (1, 2)),
#             'vect__max_df': (0.50, 0.75, 1.0),
#             'vect__max_features': (None, 5000, 10000),
            'model__estimator__max_depth':[5,6,7],
            'model__estimator__n_estimators': [400, 800, 1000],
            'model__estimator__min_samples_split': [4, 6, 8],
             }

    # perform Grid Search Cross Validation to find best parameters
    cv = GridSearchCV(estimator = pipeline, 
                      param_grid=parameters,
                      n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
      Function that checks the performance of 'model' on the test set,
    by printing the classification_report for each target of the model.
      Arguments:
      --model: the model to be evaluated
      --X_test: the input test set 
      --Y_test: the real outcomes
      --category_names: the names of the targets      
    """
    # make predictions on the test set
    Y_pred = model.predict(X_test)
    # transform to dataframe the predictions 
    Y_pred_pd = pd.DataFrame(Y_pred)
    # name the targets
    Y_pred_pd.columns = category_names
    # evaluate the predictions
    for col in category_names:  
        print(col + '\n')
        print(classification_report(Y_pred_pd[col], Y_test[col]))

    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    """save model in pickle format to path model_filepath"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
      Main function that calls the functions above:
      --load the data from the database_filepath
      --split the data into train/test
      --build the model on the train set
      --evaluate the model on the test set
      --save the model in pickle format
      Arguments:
      --database_filepath: the path to the data
      --model_filepath: the path to save the model  
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train[:100000], Y_train[:100000]) #https://github.com/scikit-learn/scikit-learn/issues/8941
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()