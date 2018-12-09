import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression 

# Figures inline and set visualization style
%matplotlib inline

trainnative = pd.read_csv(r"C:\Users\Firoz Jaipuri\Downloads\AV_DataSets\LoanPrediction\train_Loanprediction.csv")
testnative = pd.read_csv(r"C:\Users\Firoz Jaipuri\Downloads\AV_DataSets\LoanPrediction\test_LoanPrediction.csv")

train = pd.read_csv(r"C:\Users\Firoz Jaipuri\Downloads\AV_DataSets\LoanPrediction\train_Loanprediction.csv")
test = pd.read_csv(r"C:\Users\Firoz Jaipuri\Downloads\AV_DataSets\LoanPrediction\test_LoanPrediction.csv")

#Change dependents from 3+ to 3
train.loc[train['Dependents'] == '3+', 'Dependents'] = '3'
trainInt = train.iloc[:,3]
trainIntDependents = pd.to_numeric(trainInt, downcast='signed')
train.drop(['Dependents'],axis=1,inplace=True)
train = pd.concat([train,trainIntDependents],axis=1)

test.loc[test['Dependents'] == '3+', 'Dependents'] = '3'
testInt = test.iloc[:,3]
testIntDependents = pd.to_numeric(testInt, downcast='signed')
test.drop(['Dependents'],axis=1,inplace=True)
test = pd.concat([test,testIntDependents],axis=1)

#Change gender to Binary 
train.loc[train['Gender'] == 'Male','Gender'] = 0
train.loc[train['Gender'] == 'Female','Gender'] = 1

test.loc[test['Gender'] == 'Male','Gender'] = 0
test.loc[test['Gender'] == 'Female','Gender'] = 1


#Change Marital status to binary
train.loc[train['Married'] == 'Yes','Married'] = 1
train.loc[train['Married'] == 'No','Married'] = 0

test.loc[test['Married'] == 'Yes','Married'] = 1
test.loc[test['Married'] == 'No','Married'] = 0

#Change Employment status to binary
train.loc[train['Self_Employed'] == 'Yes','Self_Employed'] = 1
train.loc[train['Self_Employed'] == 'No','Self_Employed'] = 0

test.loc[test['Self_Employed'] == 'Yes','Self_Employed'] = 1
test.loc[test['Self_Employed'] == 'No','Self_Employed'] = 0

#Change Education status to binary
train.loc[train['Education'] == 'Graduate','Education'] = 1
train.loc[train['Education'] == 'Not Graduate','Education'] = 0

test.loc[test['Education'] == 'Graduate','Education'] = 1
test.loc[test['Education'] == 'Not Graduate','Education'] = 0

#Change Loan_Status status to binary for train dataset
train.loc[train['Loan_Status'] == 'Y','Loan_Status'] = 1
train.loc[train['Loan_Status'] == 'N','Loan_Status'] = 0

# OneHot encodnig for Property_Area
train = pd.concat([train,pd.get_dummies(train['Property_Area'], prefix='Property_Area')],axis=1)
train.drop(['Property_Area'],axis=1, inplace=True)

test = pd.concat([test,pd.get_dummies(test['Property_Area'], prefix='Property_Area')],axis=1)
test.drop(['Property_Area'],axis=1, inplace=True)

train['Gender'] = train['Gender'].astype(np.float64)
train['Education'] = train['Education'].astype(np.int64)
train['Married'] = train['Married'].astype(np.float64)
train['Self_Employed'] = train['Self_Employed'].astype(np.float64)

test['Gender'] = test['Gender'].astype(np.float64)
test['Education'] = test['Education'].astype(np.int64)
test['Married'] = test['Married'].astype(np.float64)
test['Self_Employed'] = test['Self_Employed'].astype(np.float64)

#EDA
test.info()
test.describe()
sns.countplot(x='Loan_Status',data=train) # the no. of approved loans are double the no. of rejected loans 

test['Loan_Status'] = 1 # create a basic model with all approved loan statuses 
test.loc[test['Loan_Status'] == 1,'Loan_Status'] = 'Y'
test.loc[test['Loan_Status'] == 0,'Loan_Status'] = 'N'
test[['Loan_ID','Loan_Status']].to_csv(r'C:\Users\Firoz Jaipuri\Downloads\AV_DataSets\LoanPrediction\SampleSubmission.csv')
