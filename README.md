# LoanPredictionAV

Analytics Vidhya Hackathon challenge - Loan Prediction 
I am using this to learn ML, data exploration and feature engineering. 

Step 1 - In the test file, all loans are approved. Gave me a score of 0.7152 on AV. 
Step 2 - After a basic exploration of the categorical variable, I found that credit history seems to be a pretty good indicator of whether loan will be approved or not. Refined base model such that in test file loan is approved if credit history is present. This increased my AV score to 0.7638. 
Step 3 - After exploratory data analysis of all the variables and imputing missign values, I build here a decision tree classifier with only 5 features.. Gender, Credit_History, LoanAmount, ApplicantIncome, SelfEmployed. The AV accuracy is 0.76388. 
