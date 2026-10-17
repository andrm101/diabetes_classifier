Overview
Diabetes is one of the most widespread chronic diseases worldwide. Early detection and prediction can support prevention, improve patient outcomes, and reduce healthcare costs. In this challenge, your task is to build a machine learning classifier that predicts whether a patient is diagnosed with diabetes or not, based on a set of health indicators.

You will be provided with a balanced subset of the CDC Diabetes Health Indicators Dataset. The dataset includes 21 features describing lifestyle, demographic, and health-related factors (e.g., BMI, smoking status, physical activity, age, blood pressure, cholesterol levels, etc.).

Your goal is to train a model that accurately classifies patients into:

Yes = Diagnosed with diabetes
No = Not diagnosed with diabetes

Submission format
ID,Diabetes
40750,No
4942,No
66565,Yes
etc.

Dataset Description
The Diabetes Health Indicators Dataset contains healthcare statistics and lifestyle survey information about people in general along with their diagnosis of diabetes. The 35 features consist of some demographics, lab test results, and answers to survey questions for each patient. The target variable for classification is whether a patient has diabetes (Yes), or healthy (No).

Files
X_train.csv - the training features
y_train.csv - the training target
X_test.csv - the test set features to be used for prediction
sample_submission.csv - a sample submission file in the correct format
Columns
HighBP - Binary - 0 = no high BP 1 = high BP
HighChol - Binary- 0 = no high cholesterol 1 = high cholesterol
CholCheck - Binary - 0 = no cholesterol check in 5 years 1 = yes cholesterol check in 5 years
BMI - Integer - Body Mass Index
Smoker - Binary - Have you smoked at least 100 cigarettes in your entire life? 0 = no 1 = yes
Stroke - Binary - (Ever told) you had a stroke. 0 = no 1 = yes
HeartDiseaseorAttack - Binary - coronary heart disease (CHD) or myocardial infarction (MI) 0 = no 1 = yes
PhysActivity - Binary - physical activity in past 30 days - not including job 0 = no 1 = yes
Fruits - Binary - Consume Fruit 1 or more times per day 0 = no 1 = yes
Veggies - Binary - Consume Vegetables 1 or more times per day 0 = no 1 = yes
HvyAlcoholConsump - Binary - Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) 0 = no 1 = yes
AnyHealthcare - Binary - Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no 1 = yes
NoDocbcCost -Binary - Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no 1 = yes
GenHlth - Integer - Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor
MentHlth - Integer - Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days
PhysHlth - Integer - Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days
DiffWalk - Binary - Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes
Gender - Binary - Female, Male