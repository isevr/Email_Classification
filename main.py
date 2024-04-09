import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')


#Utility function for preprocessing
def preprocess(text):
    text = text.lower()
    t = re.sub(r'\W+',r' ',text)
    t = re.sub(r'_',r'',t)
    t = re.sub(r'\d',r'',t)
    t = re.sub(r'\s+\S\s+',r'',t)
    t = re.sub(r'(www|com|org|list|net|mailto|subject|http)', r'', t)
    t = re.sub(r'\b\w{1,3}\b', r'', t)
    stopwords_list = stopwords.words('english') 
    txt = ' '.join([word for word in t.split() if word not in stopwords_list])
    return txt

def train_test(train,test):
    #load data
    df_train_early = pd.read_csv(train)
    df_test = pd.read_csv(test)
    df_train_early.drop_duplicates(inplace=True)
    undersample_ham = df_train_early[df_train_early.label == 'ham'].sample(frac=0.3)
    spam_df = df_train_early[df_train_early.label == 'spam']
    df_train = pd.concat([undersample_ham, spam_df]).reset_index(drop=True)
    
    
    #preprocess
    lemmatizer = WordNetLemmatizer()
    df_train['pre_email'] = [' '.join([lemmatizer.lemmatize(preprocess(email))])
                 .strip() for email in df_train['email']]
    df_test['pre_email'] = [' '.join([lemmatizer.lemmatize(preprocess(email))])
                 .strip() for email in df_test['email']]
    
    #vectorize
    tfidf= TfidfVectorizer(sublinear_tf=True, max_df=0.5, ngram_range=(1,2),
                       stop_words='english',norm='l2',binary=False,smooth_idf=False)
    
    tfidf.fit(df_train.pre_email)
    
    X_train = tfidf.transform(df_train.pre_email)
    X_train.data**=0.5
    normalize(X_train,copy=False)
    
    X_test = tfidf.transform(df_test.pre_email)
    X_test.data**=0.5
    normalize(X_test,copy=False)
    
    #fitting model
    svc = SVC(C=1,kernel='linear')
    svc.fit(X_train,df_train.label)

    
    #predictions
    y_pred = svc.predict(X_test)
    
    with open('predictions.txt', 'w', encoding='utf-8') as file:
        for prediction in y_pred:
            file.write(prediction + '\n')
    
    

train_test('train.csv','test.csv')