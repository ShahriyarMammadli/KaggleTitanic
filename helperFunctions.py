# Shahriyar Mammadli
# Kaggle Titanic competition solution, helper func
# Import required libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
# Encode dataframe
def encodeDf(df, vars):
    for var in vars:
        # Join the encoded df
        df = df.join(pd.get_dummies(df[var], prefix=var))
    # Drop columns as they are now encoded
    return df.drop(vars, axis=1)

# Analyze ticket categories
def analyzeTicket(df):
    # Extract ticket type by splitting the string by space
    return df["Ticket"].map(lambda i: 'ordinary' if len(i.split(" ")) == 1 else i.split(" ")[0])
# Confusion Matrix
def calculateAccuracy(actual, predicted):
    return accuracy_score(actual, predicted)

# Modelling using SVM
def svmModel(trainDf, testDf, targetVar):
    clf = svm.SVC(kernel='linear')
    # Fit data
    clf = clf.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return clf, clf.predict(testDf.drop(targetVar, 1))

# Random Forest Model
def rfModel(trainDf, testDf, targetVar):
    rf = RandomForestClassifier(n_estimators=100)
    # Fit data
    rf = rf.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return rf, rf.predict(testDf.drop(targetVar, 1))

# Gradient Boosting Model
def gbModel(trainDf, testDf, targetVar):
    gb = GradientBoostingClassifier(n_estimators=100, max_features='sqrt')
    # Fit data
    gb = gb.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return gb, gb.predict(testDf.drop(targetVar, 1))

# Number to Categories for Age
def categorizeAge(df):
    return df["Age"].map(lambda i: toAgeGroup(i))

# Assign age group
def toAgeGroup(age):
    if pd.isnull(age):
        return age
    elif int(age) <= 12:
        return "0-12"
    elif int(age) > 12 and int(age) <= 17:
        return "13-24"
    elif int(age) > 17 and int(age) <= 24:
        return "18-24"
    elif int(age) > 24 and int(age) <= 34:
        return "25-34"
    elif int(age) > 34 and int(age) <= 44:
        return "35-44"
    elif int(age) > 44 and int(age) <= 54:
        return "45-54"
    elif int(age) > 54 and int(age) <= 64:
        return "55-64"
    elif int(age) > 64 and int(age) <= 74:
        return "65-74"
    elif int(age) > 74:
        return "75+"
    else:
        return ""

# Build a model to predict age
def predictAge(df):
    # Categorizing the age
    df['Age'] = categorizeAge(df)
    # Drop Survived variable to avoid bias
    dataTemp = df[df['Age'].notna()].drop('Survived', 1)
    # To assess the model split non-NA samples as train and test
    X_train, X_test, y_train, y_test = train_test_split(dataTemp.drop(['Age'], 1), dataTemp['Age'],
                                                        test_size=0.3,
                                                        random_state=186)
    X_train['Age'] = y_train
    X_test['Age'] = y_test
    model, predicted = rfModel(X_train, X_test, 'Age')
    pd.DataFrame({'act': y_test, 'pred': predicted}).to_csv("AgeActualvsPredicted.csv", header=True, index=False)
    # for row in df.itertuples(index=True, name='Pandas'):
    #     if(pd.isnull(row.Age)):
    #         print(model.predict(row.drop(['Age', 'Survived'], 1)))
    return df.apply(lambda i: updateSample(i, model) if pd.isnull(i['Age']) else i, axis=1)

# Update sample by predicting Age
def updateSample(i, model):
    i['Age'] = model.predict(i.drop(['Survived', 'Age']).values.reshape(1, -1))[0]
    return i

# Voting Results i.e. Ensemble result of models
def ensembleRes(df):
    return df.apply(lambda i: helperER(i), axis=1)

# Helper function for ensembleRes() funciton
def helperER(i):
    # Take the median value which will give the result of most frequent value
    i['Ensemble'] = sorted(i.values[0:3])[1]
    return i

# Processing the name and extracting the titles
def processName(df):
    return df['Name'].apply(lambda i: i.split(", ")[1].split(" ")[0])

# Processing Cabin by taking the first charachters from the Cabin names which...
# ...implies the categories. For NAs we will use O as Other
def processCabin(df):
    return df['Cabin'].apply(lambda i: "O" if pd.isnull(i) else i[0])