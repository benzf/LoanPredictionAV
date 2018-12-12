# LoanPredictionAV

Analytics Vidhya Hackathon challenge - Loan Prediction 
I am using this to learn ML, data exploration and feature engineering. 

Step 1 - In the test file, all loans are approved. Gave me a score of 0.7152 on AV. 
Step 2 - After a basic exploration of the categorical variable, I found that credit history seems to be a pretty good indicator of whether loan will be approved or not. Refined base model such that in test file loan is approved if credit history is present. This increased my AV score to 0.7638. 
Step 3 - After exploratory data analysis of all the variables and imputing missing values, I build here a decision tree classifier with only 5 features.. Gender, Credit_History, LoanAmount, ApplicantIncome, SelfEmployed. The AV accuracy is 0.76388. 

The accuracy score  I got while attempting this problem by myself was 0.76388. The accuracy score I got after using this following tutorial was 0.77777. 

LoanPredictionAV_UsingTutorial.py  - This is code I got from AnalyticsVidhya's tutorial for the LoanPrediction problem. i have uploaded here for my own future reference.  It is a step by step approach towards a machine learning problem. It shows techniques for data exploration, visualisation and implements 4 algorithms for this problem of loan prediction. 