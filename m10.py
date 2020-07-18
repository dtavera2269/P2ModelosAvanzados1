#!/usr/bin/python

import pandas as pd
from sklearn.externals import joblib
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))




def valores(plot1): 
    

    clf = joblib.load(os.path.dirname(__file__) + '\\movie.pkl')
    vect1 = joblib.load(os.path.dirname(__file__) + '\\vect2.pkl')
    
    X_test_dtm = vect1.transform(['plot1'])
    
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family','p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance', 'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    
    y_pred_test_genres = clf.predict_proba(X_test_dtm)

    res = pd.DataFrame(y_pred_test_genres,columns=cols)
    
    str_result=''
    for c in res.columns:
        if res[c][0]>0.5:
             str_result=str_result + c
    
    return str_result


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add a plot')
        
    else:

        url = sys.argv[1]

        p1 = predict_proba(plot)
        
        print(plot)
        print('predict genre: ', p1)