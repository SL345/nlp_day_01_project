# %load q02_tokenize/build.py
# Default imports

from nltk.tokenize import TreebankWordTokenizer

import pandas as pd

from greyatomlib.nlp_day_01_project.q01_load_data.build import q01_load_data

# Write your solution here:
def q02_tokenize(path):
    df,X_train,X_test,y_train,y_test = q01_load_data(path)
    x_train_ser = pd.Series(X_train)
    x_train_lower_case = x_train_ser.str.lower()
    tokenizer = TreebankWordTokenizer()
    variable = x_train_lower_case.apply(lambda row: tokenizer.tokenize(str(row)))
    return variable

