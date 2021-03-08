# manipulate dataframes
import pandas as pd
import datatable as dtbl
#numerics
import numpy as np
#dates
import datetime as dt
# garbage collection
import gc
#notebook displays
from IPython.display import display_html, Image

# visuals
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo
# missing values
import missingno as msno
# PCA, clustering
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

#maps
import folium
from folium.plugins import HeatMap
#wordcloud
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import wordpunct_tokenize # function to split up our words

#graphs
import networkx as nx
#normality tests
import scipy.stats as st
from scipy.stats import shapiro, normaltest, anderson, chisquare, jarque_bera, kstest, f_oneway
from statsmodels.stats.diagnostic import lilliefors

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

pd.options.display.max_rows = 1000

def value_cnts(df, cat_vars):
    '''returns value counts of categorical variables cat_vars(one or many)'''
    df_cnt = df[cat_vars].value_counts().reset_index().sort_values(by = 0, ascending=False).rename(columns={0:'count'})
    df_cnt['% of total'] = round(df_cnt['count']/len(df)*100, 2)
    return df_cnt

def display_side_by_side(*args):
    '''diplay two or more dataframes side-by-side'''
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
    

## find NAs  
def inspect_missing_data(df):  
    '''find NAs for each column of a dataframe'''
    '''return head&tail of columns with NAs'''
    missing_data = df.isna().sum().reset_index().sort_values(by=0, ascending=False)
    no_missing = missing_data[missing_data[0] != 0].shape[0]
    total_cols = df.shape[1]
    total_rows = df.shape[0]
    
    missing_data.columns = ["name", "missing appearences"]
    missing_data["%missing from total"] = (missing_data[missing_data["missing appearences"]!=0]["missing appearences"]/total_rows)*100
    
    too_much_miss = missing_data[missing_data["%missing from total"] > 80].shape[0]
    to_drop = missing_data[missing_data["%missing from total"] > 80]["name"].to_list()
    
    print("There are {}/{} columns with missing data.".format(no_missing, total_cols))
    print("There are {}/{} columns with more than 80% missing data".format(too_much_miss, no_missing))
    print("Features with largest/smallest percent of missing values(top 10): ")
    
    tail = round(missing_data.tail(10).sort_values(by='%missing from total'), 2)
    head = round(missing_data.head(10).sort_values(by='%missing from total'), 2) 
       
    return display_side_by_side(head, tail)


## some colors    
bo = ["#6930c3","#5e60ce","#0096c7","#48cae4","#ade8f4","#ff7f51","#ff9b54","#ffbf69"]
pink = ["#aa4465","#dd2d4a","#f26a8d","#f49cbb","#ffcbf2","#e2afff","#ff86c8","#ffa3a5","#ffbf81","#e9b827","#f9e576"]    


def custom_palette(custom_colors):
    '''display color pallete'''
    customPalette = sns.set_palette(sns.color_palette(custom_colors))
    sns.palplot(sns.color_palette(custom_colors),size=0.8)
    plt.tick_params(axis='both', labelsize=0, length = 0)  
    
seed=42

def correlation_network(data, corr_interval=[-0.7, 0.7],
                        node_color="blue"):
    '''print most correlated features (outside [-0.7, 0.7]) into a graph'''
    
    corr = data.corr()
    corr = pd.melt(corr.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']

    # Filter only high correlations
    high_corr = corr[((corr['value'] <= corr_interval[0]) | (corr['value'] >= corr_interval[1]))]
    high_corr = high_corr[(high_corr['value'] != 1)].reset_index(drop=True)
    # the graph  
    sources = list(high_corr.x.unique())
    targets = list(high_corr.y.unique())
    
    plt.figure(figsize=(20, 10))
    g = nx.from_pandas_edgelist(high_corr, source='x', target='y') 
    layout = nx.spring_layout(g, iterations=90, k=1, seed=seed)
    target_size = [g.degree(t) * 80 for t in targets]
    nx.draw_networkx_nodes(g, 
                           layout, 
                           nodelist=targets, 
                           node_size=target_size, 
                           node_color=node_color)
    # Draw connections
    nx.draw_networkx_nodes(g, layout, nodelist=sources, node_color=node_color, node_size=800, alpha=0.5)
    nx.draw_networkx_edges(g, layout, width=1, edge_color="lime")
    target_dict = dict(zip(targets, targets))
    nx.draw_networkx_labels(g, layout, labels=target_dict)

    plt.axis('off')
    plt.title("Highest Correlations Network (outside the [-0.7, 0.7] interval)", fontsize=20)
    plt.show();
    
    
def anovaf(frame, cat_vars):
    '''check with ANOVA tests the hypothesis that two or more groups have the same population mean'''
    '''the test is applied to samples from two or more groups, possibly with differing sizes'''
    '''the samples are the value groups of a categorical variable'''
    '''return p-values of tests'''
    anv = pd.DataFrame()
    anv['feature'] = cat_vars ## list of categorical variables of interest
    pvals = []
    for c in cat_vars:
        samples = []
        for cls in frame[c].unique(): ## unique values of the categorical variable
            s = frame[frame[c] == cls]['price'].values ##save prices per value of category
            samples.append(s)
        pval = f_oneway(*samples)[1] ## ANOVA test
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval') ## return p-values


def encode(frame, feature):
    '''encode categorical variable by mean price of category values'''
    '''return new encoded features'''
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'price']].groupby(feature).mean()['price']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o    
    
    
    
def spearman(frame, features, target):
    '''barplot of Spearman's correlation coefficient for all features with respect to the target'''
    ## create correlation coefficients
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame[target], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    ## plot data
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    plt.tick_params(axis='both', labelsize=12)
    plt.ylabel('')
    plt.title('Spearman correlations for all features with respect to ' + target + '\n', weight="bold", fontsize=16);
    plt.show()    
        
        
        
    
    