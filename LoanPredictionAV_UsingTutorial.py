import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline 
import warnings 
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


train = pd.read_csv(r"C:\Users\Firoz Jaipuri\DS_Projects\AV_DataSets\LoanPrediction\train_LoanPrediction.csv")
test = pd.read_csv(r"C:\Users\Firoz Jaipuri\DS_Projects\AV_DataSets\LoanPrediction\test_LoanPrediction.csv")

train_original = train.copy()
test_original = test.copy()

#### Understanding the Data 
train.columns
test.columns
train.dtypes 
train.shape
test.shape 

#Univariate Analysis 
train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts(normalize=True)
train['Loan_Status'].value_counts().plot.bar() #Loan of abbout 422(70%) people was approved 

### Categorize Variables ###
# 1. Categorical Variables - Gender, Married, Self_Employed, Credit_History, Loan_Status
# 2. Ordinal variables - Dependents, Education, Property_Area
# 3. Numerical variables - ApplicantIncome, CoapplicantIncome, LoanAmount, LoanAmountTerm

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Gender') # 80% applicants are male

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married') #ard 65% applicatnts are married

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')# ard 15% are self_employed

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')# ard 85% have good credit history

plt.show()

plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents') #most applicants dont have dependents

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') # ard 80% applicants are graduate

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')# most are from semiurban area 
plt.show()


plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']) # not normally distributed 

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5)) # outliers are present in applicantIncome
plt.show()

train.boxplot(column='ApplicantIncome',by='Education') # graduates have higher incomes and outliers too 
plt.suptitle("")
plt.show()

plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);

plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()

plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(df['LoanAmount']); #loanamount if fairly normally distributed. 

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
plt.show() 


########### BiVariate Analysis

### Categorical and Target Variables 

Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))#Loan approval seems to be Gender agnostic 

Married=pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() #Proportion of married applicants is higher for the approved loans.

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show() #Distribution of applicants with 1 or 3+ dependents is similar across both the categories of Loan_Status.

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() #Loan approval seems to be Education agnostic 


Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() # Loan approval seems to be Employment type agnostic 

Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() #It seems people with credit history as 1 are more likely to get their loans approved.

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show() #Proportion of loans getting approved in semiurban area is higher as compared to that in rural or urban areas.


##### Numerical and Target Variables 
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar() # shows no difference between teh mean incomes of those with approved loans and those with not approved loans 

bins=[0, 2500,4000,6000,81000]
group = ['Low','Average','High','VeryHigh']
train['Income_bin'] = pd.cut(df['ApplicantIncome'],bins,labels=group)
Income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis =0).plot(kind="bar", stacked = True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage') #Applicant income does not affect the chances of loan approval 

bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(df['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('CoapplicantIncome')
P = plt.ylabel('Percentage')

# if coapplicant’s income is less the chances of loan approval are high. But this does not look right. 
# The possible reason behind this may be that most of the applicants don’t have any coapplicant so the 
# coapplicant income for such applicants is 0 and hence the loan approval is not dependent on it. So we can 
# make a new variable in which we will combine the applicant’s and coapplicant’s income to visualize the 
# combined effect of income on loan approval.

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)

Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage') # Proportion of loans getting approved for applicants having low Total_Income is very less as compared to others

bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(df['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')

#drop the bins created for exploration part 
train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)

train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+', 3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)

matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
# most correlated variables are (ApplicantIncome - LoanAmount) and (Credit_History - Loan_Status). 
# LoanAmount is also correlated with CoapplicantIncome.


############Missing Value Imputation 
train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


######### Outlier Treatment 

# Due to these outliers bulk of the data in the loan amount is at the left and the right tail is longer. This is called right
# skewness. One way to remove the skewness is by doing the log transformation. As we take the log transformation, 
# it does not affect the smaller values much, but reduces the larger values. So, we get a distribution similar to 
# normal distribution.

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])


############## Feature Engineering #################
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']

#sns.distplot(train['Total_Income']);
train['Total_Income_log'] = np.log(train['Total_Income'])
#sns.distplot(train['Total_Income_log']);
test['Total_Income_log'] = np.log(test['Total_Income'])

train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']

#sns.distplot(train['EMI']);

train['Balance Income']=train['Total_Income']-(train['EMI']*1000) # Multiply with 1000 to make the units equal 
test['Balance Income']=test['Total_Income']-(test['EMI']*1000)

#drop the variables which we used to create these new features. else correlation between these old and new features will be very high
train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)


############################################################################################################################
####### Model Building 

train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)
X = train.drop('Loan_Status',1)
y = train.Loan_Status

X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

# model = LogisticRegression()
# model.fit(x_train, y_train)
# pred_cv = model.predict(x_cv)
# print(accuracy_score(y_cv,pred_cv))
# pred_test = model.predict(test)

# submission=pd.read_csv(r"C:\Users\Firoz Jaipuri\DS_Projects\AV_DataSets\LoanPrediction\Sample_Submission_ZAuTl8O_FK3zQHh.csv")

########## Logistic Regression with k-fold startification ###########
# i=1
# kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
# for train_index,test_index in kf.split(X,y):
#      #print('\n{} of kfold {}'.format(i,kf.n_splits))
#      xtr,xvl = X.loc[train_index],X.loc[test_index]
#      ytr,yvl = y[train_index],y[test_index]
    
#      model = LogisticRegression(random_state=1)
#      model.fit(xtr, ytr)
#      pred_test = model.predict(xvl)
#      score = accuracy_score(yvl,pred_test)
#      #print('accuracy_score',score)
#      i+=1
# pred_test = model.predict(test)
# pred=model.predict_proba(xvl)[:,1]
#######################################################################

########## Decision tree  with k-fold startification ##################
# i=1
# kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
# for train_index,test_index in kf.split(X,y):
#      print('\n{} of kfold {}'.format(i,kf.n_splits))
#      xtr,xvl = X.loc[train_index],X.loc[test_index]
#      ytr,yvl = y[train_index],y[test_index]
    
#      model = tree.DecisionTreeClassifier(random_state=1)
#      model.fit(xtr, ytr)
#      pred_test = model.predict(xvl)
#      score = accuracy_score(yvl,pred_test)
#      print('accuracy_score',score)
#      i+=1
# pred_test = model.predict(test)
#######################################################################

########## Random forest with k-fold startification ##################
# i=1
# kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
# for train_index,test_index in kf.split(X,y):
#      print('\n{} of kfold {}'.format(i,kf.n_splits))
#      xtr,xvl = X.loc[train_index],X.loc[test_index]
#      ytr,yvl = y[train_index],y[test_index]
    
#      model = RandomForestClassifier(random_state=1, max_depth=10)
#      model.fit(xtr, ytr)
#      pred_test = model.predict(xvl)
#      score = accuracy_score(yvl,pred_test)
#      print('accuracy_score',score)
#      i+=1
# pred_test = model.predict(test)

# # Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators
# paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
# grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)
# x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state=1)
# # Fit the grid search model
# grid_search.fit(x_train,y_train)

# GridSearchCV(cv=None, error_score='raise',
#        estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#             oob_score=False, random_state=1, verbose=0, warm_start=False),
#        fit_params=None, iid=True, n_jobs=1,
#        param_grid={'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'n_estimators': [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]},
#        pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
#        scoring=None, verbose=0)

# # Estimating the optimized value
# grid_search.best_estimator_

# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=3, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=41, n_jobs=1,
#             oob_score=False, random_state=1, verbose=0, warm_start=False)
# i=1
# kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
# for train_index,test_index in kf.split(X,y):
#      print('\n{} of kfold {}'.format(i,kf.n_splits))
#      xtr,xvl = X.loc[train_index],X.loc[test_index]
#      ytr,yvl = y[train_index],y[test_index]
    
#      model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)
#      model.fit(xtr, ytr)
#      pred_test = model.predict(xvl)
#      score = accuracy_score(yvl,pred_test)
#      print('accuracy_score',score)
#      i+=1
# pred_test = model.predict(test)
# pred2=model.predict_proba(test)[:,1]

#######################################################################

########## XGBoost with k-fold startification ##################

i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = XGBClassifier(n_estimators=50, max_depth=4)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
pred3=model.predict_proba(test)[:,1]

#######################################################################



# fpr, tpr, _ = metrics.roc_curve(yvl,  pred)
# auc = metrics.roc_auc_score(yvl, pred)
# plt.figure(figsize=(12,8))
# plt.plot(fpr,tpr,label="validation, auc="+str(auc))
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc=4)
# plt.show()
submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']
submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv(r'C:\Users\Firoz Jaipuri\DS_Projects\AV_DataSets\LoanPrediction\Logistic.csv')

importances=pd.Series(model.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(12,8))