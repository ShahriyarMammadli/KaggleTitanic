# Shahriyar Mammadli
# Kaggle Titanic competition solution
# Import required libraries
import pandas as pd
from sklearn import svm
import helperFunctions as hf

# Train dataframe data preparation
trainDf = pd.read_csv('data/train.csv')
# See the columns in the dataframe
print("-------------- Column names: --------------")
print(trainDf.columns)
print("------------------ Head: ------------------")
print(trainDf.head())
# Return the number of NAs for each column
print("-------------- Number of NAs: --------------")
print(trainDf.isna().sum())
# The number of unique values for each column
print("--------- Number of unique values: ---------")
print(trainDf.nunique())
print("------------------ Shape: ------------------")
print(trainDf.shape)
# Embarked has only two missing values, so drop NAs for that column
trainDf = trainDf[trainDf['Embarked'].notna()]
print("------------------ Shape: ------------------")
print(trainDf.shape)
# Cabin has too many missing values, so drop that column
trainDf.drop('Cabin', axis=1, inplace=True)
print("------------------ Shape: ------------------")
print(trainDf.shape)
# Drop NAs for age column
trainDf = trainDf[trainDf['Age'].notna()]
# Drop id and name columns
trainDf.drop(['PassengerId', 'Name', 'Fare', 'Age'], axis=1, inplace=True)
print("------------------ Shape: ------------------")
print(trainDf.shape)
# Process Ticket since it has different prefixes for different types
trainDf[["Ticket"]] = hf.analyzeTicket(trainDf[["Ticket"]])
print("------------------ Shape: ------------------")
print(trainDf.shape)
# Encode dataframe
# trainDf = hf.encodeDf(trainDf, ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked', 'Ticket'])
print("-------------- Column names: --------------")
print(trainDf.columns)

# Test dataframe data preparation
testDf = pd.read_csv('data/test.csv')
originalDf = testDf
# Read actual results
actualDf = pd.read_csv('data/gender_submission.csv')
testDf = pd.merge(testDf, actualDf, left_on='PassengerId', right_on='PassengerId')
print("------------------ Shape: ------------------")
print(testDf.shape)
print("-------------- Number of NAs: --------------")
print(testDf.isna().sum())
# Embarked has one missing values, so drop NAs for that column
#testDf = testDf[testDf['Fare'].notna()]
# Drop id and name columns, Cabin
testDf.drop(['PassengerId', 'Name', 'Cabin', 'Fare', 'Age'], axis=1, inplace=True)
# Drop NAs for age column
# testDf = testDf[testDf['Age'].notna()]
print("------------------ Shape: ------------------")
print(testDf.shape)
# Process Ticket since it has different prefixes for different types
testDf[["Ticket"]] = hf.analyzeTicket(testDf[["Ticket"]])
print("------------------ Shape: ------------------")
print(testDf.shape)
# Encode dataframe
# testDf = hf.encodeDf(testDf, ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked', 'Ticket'])
print("------------------ Shape: ------------------")
print(trainDf.shape)
print("------------------ Shape: ------------------")
print(testDf.shape)
# Since some of the categorical values may exist in test but not in train...
# ...before encoding they needed to ber merged, and splitting again after encoding
mergedDf = pd.concat([trainDf, testDf],ignore_index=True)
mergedDf = hf.encodeDf(mergedDf, ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked', 'Ticket'])
trainDf = mergedDf[0:trainDf.shape[0]]
testDf = mergedDf[trainDf.shape[0]:mergedDf.shape[0]]
print("------------------ Shape: ------------------")
print(trainDf.shape)
print("------------------ Shape: ------------------")
print(testDf.shape)

# Modelling
# SVM model
# Initialize SVM classifier
clf = svm.SVC(kernel='linear')
# Fit data
clf = clf.fit(trainDf.drop('Survived', 1), trainDf[['Survived']])
predicted = clf.predict(testDf.drop('Survived', 1))
originalDf[['Survived']] = predicted
print(hf.calculateAccuracy(testDf['Survived'],predicted))
originalDf[['PassengerId', 'Survived']].to_csv("submission.csv", header=True, index=False)
# TODO: user Age and Fare
# TODO: Do ensemble