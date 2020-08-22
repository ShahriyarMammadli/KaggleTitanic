# KaggleTitanic
This solution had **0.79186** score which is in top **2k** in the leaderboard. 

## Requirements
The list of the libraries that needed to be installed are:  
* **pandas**
* **scikit-learn**

## Approach
The dataset is analyzed in detail to have a deep understanding of it. Variables are cleansed from the problems and new features are created from the existing ones. Then single algorithms and ensemble techniques are applied to data.

### Exploratory Data Analysis(EDA) and Feature Engineering
**Step 1:** Number of missing values
Variables with missing values will require special treatment, they will either be predicted by modeling or approximations will be given by simple techniques (e.g. mean, median, etc.). Decisions will be made depending on the variable type and number of missing values. 

_Train Set_  
* There are 177 missing values for Age
* There are 687 missing values for Cabin  
* There are 2 missing values for Embarked 

_Test Set_  
* There are 86 missing values for Age
* There are 327 missing values for Cabin 
* There are 1 missing values for Fare

**Step 2:** Number of unique values  
By analyzing the number of unique values continues and categorical values can be observed. Thus, categorical variables can be chosen which will be one-hot-encoded (unless categories have naturally ordered relationship).    
| Variable | # of unique values  |
| :-----: | :-: |
| PassengerId | 891 |
| Survived | 2 |
| Pclass | 3 |
| Name | 891 |
| Sex | 2 |
| Age | 88 |
| SibSp | 7 |
| Parch | 7 |
| Ticket | 681 |
| Fare | 248 |
| Cabin | 147 |
| Embarked | 3 |  

**Step 3:** Processing _Ticket_ variable
The ticket has types like A/4, PC, etc. and ordinary tickets which only consisted of numbers. Those prefixes are extracted from the samples to specify categories of tickets and the ones without a prefix are categorized as an "ordinary" ticket.  

**Step 4:** Processing _Name_ variable
The names have titles like Ms., Mrs., Don, etc. Those are extracted from the names and a new variable is created by using these titles.

**Step 5:** Processing _Cabin_ variable
The Cabin variable has many missing values. It seems hard to fill these values by building a model or using any other technique since it has a small number of existing samples. However, existing samples may deliver enough information. Additionally, maybe the absence of the other samples may also have some information on it. Thus we can fill the NA samples with 'other' or 'unknown'. Existing Cabin samples indicate that cabins are named with a letter and following number e.g. A1, C3, etc. Thus, we categorize the existing samples with the first letter of cabin values and use 'o' (meaning other) for missing values.

**Step 7:** Processing _Embarked_ variable
The Embarked variable has two missing values, those samples are dropped for now. 

**Step 8:** Processing _PassengerId_ variable
The PassengerId variable is eliminated since it does not carry any information.

**Step 9:** Processing _Fare_ variable
The Fare variable has a single missing value in the test set. This sample is filled by using the mean of the other samples from the test set.

**Step 10:** Processing _Age_ variable
The Age variable may carry important information about survival. It has **(177+86)/1309 ≈ 20%** (NAs for Train set + NAs for Test set)/Total number of samples. Thus, it is not logical to fill missing values by simple techniques. Before, modeling for survival, a model may be built for Age. From existing samples, train and test sets are created by 0.7 and 0.3 ratios respectively. Using all variables, except PassengerId and Survived model build to predict age category of missing samples. However, before this step instead of building a regression model, I decided to use categorical age groups. For this purpose, most common age groups are used.

| Age group |
| :-----: |
| 0-12 |
| 13-24 |
| 25-34 |
| 35-44 |
| 45-54 |
| 55-64 |
| 64-75 |
| 75+ |

After converting ages into the above-mentioned groups, the Random Forest model is built to predict age groups. With the help of test samples, it is observed that predictions are good enough to use this model because predicted categories are very close to actual categories. 

**Step 11:** One-hot-encoding
Before starting the modeling process one-hot-encoding is applied to the variables Sex, Pclass, SibSp, Parch, Embarked, Ticket, Name, Cabin. [Dropping the one variable from resulting one-hot-encoding](https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/) is tried but it decreased the accuracy slightly thus, it is not used in the final model.


## Modelling
Two techniques are used for modeling one is using single algorithms and the second is to combine those models to put forward an ensemble model. **SVM**, **Random Forest**, **Gradient Boosting** models are used for modeling purposes. Also, an ensemble model is built by using these three models' median results (since the median result of these three models will be the most frequent output for Survived).

## Result
| Model | Score  |
| :-----: | :-: |
| SVM | 0.76555 |
| RF | 0.74401 |
| GB | 0.79186 |
| Ensemble | 0.77272 |

As a result, GB performed best with parameters of _n_estimators=100, max_features='sqrt'_. The why of the ensemble model performed worse than GB is due to SVM and RF were mostly failed together to predict for Survived for a sample. 

# ToDo
In total 7-8 hours are spent on building this model, thus, it can be improved by putting additional effort. While analyzing the data, it was obvious that all data samples and features have some information, therefore, all of them must be somehow used to obtain maximum accuracy. Some of the possible improvements are:
* Predicting the missing samples of Embarked variable
* Tuning the hyperparameters
* Eliminating the '.' and '/' characters from the ticket samples (I noticed it after submitting the result)
* Using other algorithms especially I am curious about CatBoost's performance.

