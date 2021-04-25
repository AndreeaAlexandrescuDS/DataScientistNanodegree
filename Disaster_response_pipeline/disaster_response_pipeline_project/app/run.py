import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
database_filepath = '/home/jovyan/work/analysis/DSNanodegree/DisasterResponseProject/data/DBAndreeaA.db'
df = pd.read_sql_table('input_disaster', 'sqlite:///' + database_filepath)

# load model
model = joblib.load("/home/jovyan/work/analysis/DSNanodegree/DisasterResponseProject/models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    medical_help_cnt = df.groupby('medical_help').count()['message']
    medical_help_names =  ["No medical help requiered', 'Medical help required']
    
    water_cnt = df.groupby('water').count()['message']
    water_names =  ['No water request', 'Water request']
    
    food_cnt = df.groupby('food').count()['message']
    food_names =  ['No food request', 'Food request']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=medical_help_names,
                    y=medical_help_cnt
                )
            ],

            'layout': {
                'title': 'Distribution of Medical help messages ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Medical help requests"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=water_names,
                    y=water_cnt
                )
            ],

            'layout': {
                'title': 'Distribution of Water requests messages ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Water requests"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=food_names,
                    y=food_cnt
                )
            ],

            'layout': {
                'title': 'Distribution of Food help messages ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Food requests"
                }
            }
        },
        
        
        
        
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()