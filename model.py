import pandas as pd
import numpy as np
loan = pd.read_csv('Loan.csv',encoding='utf-8')

loan.Gender = loan.Gender.fillna('Male')
loan.Married=loan.Married.fillna('Yes')
loan.Dependents=loan.Dependents.fillna('0')
loan.Self_Employed=loan.Self_Employed.fillna('No')
loan.LoanAmount=loan.LoanAmount.fillna(loan.LoanAmount.mean())
loan.Loan_Amount_Term=loan.Loan_Amount_Term.fillna('360.0')
loan.Credit_History=loan.Credit_History.fillna('1.0')

loan.Credit_History=loan.Credit_History.fillna('1.0')
loan.Education=loan.Education.replace({'Graduate':1,'Not Graduate':0})
loan.Self_Employed=loan.Self_Employed.replace({'Yes':1,'No':0})
loan.Dependents=loan.Dependents.replace({'3+':3})
loan.Property_Area=loan.Property_Area.replace({'Urban':2,'Rural':0,'Semiurban':1})
loan.Loan_status=loan.Loan_status.replace({'Y':1,'N':0})

loan = loan.drop(['Loan_ID'],axis=1)

temp = loan
temp.ApplicantIncome = pd.DataFrame(temp.ApplicantIncome)

y = -1
for x in temp.ApplicantIncome:
    y = y+1
    if (x > 0 and x < 5000):
        temp.ApplicantIncome[y] = 0

y = -1
for x in temp.ApplicantIncome:
    y = y+1
    if (x >= 5000 and x < 10000):
        temp.ApplicantIncome[y] = 1

y = -1
for x in temp.ApplicantIncome:
    y = y+1
    if  x >= 10000:
        temp.ApplicantIncome[y] = 2

y = -1
for x in temp.CoapplicantIncome:
    y = y + 1
    if x > 0:
        temp.CoapplicantIncome[y] = 1

temp.Loan_Amount_Term = temp.Loan_Amount_Term.astype(float)
loan.Gender=loan.Gender.replace({'Male':1,'Female':0})
loan.Married=loan.Married.replace({'Yes':1,'No':0})

# splitting training and test data
from sklearn.model_selection import train_test_split

# y = f(x)
X = temp[temp.columns[:-1]] # all but last column --- contains feature columns
Y = temp[temp.columns[-1]]  # last column  --- contains target column

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=50)
X_train.shape
Y_train.shape
X_test.shape
Y_test.shape

from sklearn.linear_model import LogisticRegression

loan_model = LogisticRegression()
# Fitting Logistic Regression to the Training set
loan_model.fit(X_train, Y_train)
Y_pred = loan_model.predict(X_test)
z = loan_model.score(X_test, Y_test)

print(z)

from joblib import dump
dump(loan_model, 'file.joblib')
from joblib import load
new_loanapp = load('file.joblib')
new_loan_application = [0, 0, 0, 0, 1, 1, 0.0, 405.0, 360.0, 0, 0]
p = new_loanapp.predict([new_loan_application])
print(p)
