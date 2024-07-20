import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
#loading csv file
df=pd.read_csv('model deployment\iris.csv')

#viewing top 5 rows
print(df.head())

#dependent and independent features
x=df.drop('Class', axis=1)
y=df['Class']

print(x.head()) #independent features
print(y.head()) #dependent features

#splitting dataset into train-test
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=50)

#feature scaling
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)

#using randomForestClassifier
ran_class=RandomForestClassifier()

#fit the model
ran_class.fit(x_train, y_train)

#creating pickle file of our model
pickle.dump(ran_class, open('MoDel.pkl', 'wb'))
