# Shahriyar Mammadli
# Kaggle Titanic competition solution
# Import required libraries
import pandas as pd
import helperFunctions as hf

# Train dataframe data preparation
trainDf = pd.read_csv('data/train.csv')
# See the columns in the dataframe
# Return the number of NAs for each column
print("--------- Number of NAs in Train Set: ---------")
print(trainDf.isna().sum())
# The number of unique values for each column
print("---- Number of unique values in Train Set: ----")
print(trainDf.nunique())
# Embarked has only two missing values, so drop NAs for that column
trainDf = trainDf[trainDf['Embarked'].notna()]
# Drop id and name columns
trainDf = trainDf.drop(['PassengerId'], axis=1)
# Process Ticket since it has different prefixes for different types
trainDf[["Ticket"]] = hf.analyzeTicket(trainDf[["Ticket"]])

# Test dataframe data preparation
testDf = pd.read_csv('data/test.csv')
originalDf = testDf
# Read actual results
actualDf = pd.read_csv('data/gender_submission.csv')
testDf = pd.merge(testDf, actualDf, left_on='PassengerId', right_on='PassengerId')
print("--------- Number of NAs in Test Set: ---------")
print(testDf.isna().sum())
# Drop id and name columns, Cabin
testDf = testDf.drop(['PassengerId'], axis=1)
# Process Ticket since it has different prefixes for different types
testDf[["Ticket"]] = hf.analyzeTicket(testDf[["Ticket"]])
# Since some of the categorical values may exist in test but not in train...
# ...before encoding they needed to ber merged, and splitting again after encoding
mergedDf = pd.concat([trainDf, testDf],ignore_index=True)
# Process Names and extract titles from them
mergedDf['Name'] = hf.processName(mergedDf)
# Fare has single missing value fill it by mean value
mergedDf.loc[mergedDf.loc[pd.isna(mergedDf['Fare']), :].index[0], 'Fare'] = testDf['Fare'].mean()
# Cabin has missing values these missing values will be filled
mergedDf['Cabin'] = hf.processCabin(mergedDf)
mergedDf = hf.encodeDf(mergedDf, ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked', 'Ticket', 'Name', 'Cabin'])
# Convert age into categories
mergedDf = hf.predictAge(mergedDf)
mergedDf = hf.encodeDf(mergedDf, ['Age'])
trainDf = mergedDf[0:trainDf.shape[0]]
testDf = mergedDf[trainDf.shape[0]:mergedDf.shape[0]]

# Modelling
# SVM model
# Initialize SVM classifier
modelSVM, predictedSVM = hf.svmModel(trainDf, testDf, 'Survived')
# Random Forest model
modelRF, predictedRF = hf.rfModel(trainDf, testDf, 'Survived')
# Gradient Boosting model
modelGB, predictedGB = hf.gbModel(trainDf, testDf, 'Survived')
ensembleDf = pd.DataFrame({'SVM': predictedSVM, 'RF': predictedRF, 'GB': predictedGB, 'Ensemble': None})
ensembleDf = hf.ensembleRes(ensembleDf)
# Votes of models
ensembleDf.to_csv("ensembleRes.csv", header=True, index=False)
originalDf[['Survived']] = predictedGB
originalDf[['PassengerId', 'Survived']].to_csv("submission.csv", header=True, index=False)

