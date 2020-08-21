# Shahriyar Mammadli
# Kaggle Titanic competition solution, helper func
# Import required libraries
import pandas as pd
from sklearn.metrics import accuracy_score

# Encode dataframe
def encodeDf(df, vars):
    for var in vars:
        # Join the encoded df
        df = df.join(pd.get_dummies(df[var], prefix=var, drop_first=True))
    # Drop columns as they are now encoded
    return df.drop(vars, axis=1)

# Analyze ticket categories
def analyzeTicket(df):
    # Extract ticket type by splitting the string by space
    return df["Ticket"].map(lambda i: 'ordinary' if len(i.split(" ")) == 1 else i.split(" ")[0])
# Confusion Matrix
def calculateAccuracy(actual, predicted):
    return accuracy_score(actual, predicted)