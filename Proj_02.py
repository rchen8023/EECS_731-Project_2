import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('Data/Shakespeare_data.csv')
# change play and player from string to numbers
data['Play'] = data['Play'].astype('category')
data['Play'] = data['Play'].cat.rename_categories(list(range(1,37))).astype('int')
data['Player'] = data['Player'].astype('category')
data['Player'] = data['Player'].cat.rename_categories(list(range(1,935))).astype('int')
data = data.dropna() # drop all Nan value

splitASL = data['ActSceneLine'].astype('str').str.split(pat='.',expand=True
splitASL = splitASL.rename(columns={0:'Act',1:'Scene',2:'Line'})
data = pd.concat([data, splitASL], axis=1, sort=False)

# remove some unused features. 
data = data.drop(['Dataline', 'ActSceneLine', 'PlayerLine'], axis=1)
data.to_csv('Data/new_datasets.csv', index=False)

label = data['Player']
sample = data.drop('Player',axis=1)
sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size=0.1)

model_1 = GaussianNB()
model_1.fit(sample_train,label_train)
label_predict_1 = model_1.predict(sample_test)
accuracy_1 = model_1.score(sample_test,label_test)

model_2 = RandomForestClassifier(n_estimators=10)
model_2.fit(sample_train,label_train)
label_predict_2 = model_2.predict(sample_test)
accuracy_2 = model_2.score(sample_test,label_test)

model_3 = DecisionTreeClassifier()
model_3.fit(sample_train,label_train)
label_predict_3 = model_3.predict(sample_test)
accuracy_3 = model_3.score(sample_test,label_test)

