import pandas as pd
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
m=LinearRegression()
df1=pd.read_excel("C:\\Users\\DELL\\Desktop\\ml\\Test_dataset.xlsx")

df=pd.read_excel("C:\\Users\\DELL\\Desktop\\ml\\Train_dataset.xlsx")

from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
df['Region']= label_encoder.fit_transform(df['Region'])
df1['Region']= label_encoder.fit_transform(df1['Region'])

l=['Gender','Designation','Name','Married','Occupation','Mode_transport','comorbidity','Pulmonary score','cardiological pressure']
for i in l:
    df[i]= label_encoder.fit_transform(df[i].astype('str'))
    df1[i]= label_encoder.fit_transform(df1[i].astype('str'))

df.fillna(df.mean(),inplace=True)
df1.fillna(df1.mean(),inplace=True)


x_train=df.drop(['Infect_Prob'],axis=1)
y_train=df['Infect_Prob']
y_train=y_train.astype('int')
xtrain,xtest,ytrain,ytest=cross_validation.train_test_split(x_train,y_train,test_size=0.3,random_state=30)


from sklearn.ensemble import RandomForestClassifier as RFC
rfc = RFC()
rfc.fit(xtrain,ytrain)
ypred1 = rfc.predict(xtest)
xc=pd.DataFrame({'Actual':ytest,'Predicted':ypred1}).head(10)

y2=rfc.predict(df1)


df1['Infect_Prob'] = pd.Series(rfc.predict(df1),index=df1.index)

d2=df1.loc[:,["people_ID","Infect_Prob"]]
d2.to_excel("C:\\Users\\DELL\\Desktop\\ml\\out1_dataset.xlsx")
