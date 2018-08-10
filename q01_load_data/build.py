# %load q01_load_data/build.py
# Default imports
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

path = 'data/20news-bydate-train/'
# Your solution here:
def q01_load_data(path,seed = 9):
    features = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
    news = load_files(path,features)
    X_train,X_test,y_train,y_test = train_test_split(news.data, news.target,test_size=0.8,random_state=seed)
    return news,X_train,X_test,y_train,y_test

