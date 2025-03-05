import pandas as ps
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import _pickle

#load CV file
df = ps.read_csv("new_file.csv")
print(df.head())

#Select independent and dependent variable
x = df.iloc[:, 1:]
y = df.iloc[:, 0]

#split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

#Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.values.reshape(-1, 1))  # ✅ Reshape for scaler
y_test = sc_y.transform(y_test.values.reshape(-1, 1))

#Instatiate the model
classifier = RandomForestRegressor(n_estimators=200, random_state=200)


#Fit The Model
classifier.fit(x_train, y_train.ravel())

accuracy = classifier.score(x_test, y_test)

#Check Accuracy
print("Accuracy:",accuracy)


#make pickle
with open("model.pkl", "wb") as model_file:
    _pickle.dump(classifier, model_file)  

with open("scaler_x.pkl", "wb") as scaler_x_file:
    _pickle.dump(sc, scaler_x_file)  

with open("scaler_y.pkl", "wb") as scaler_y_file:
    _pickle.dump(sc_y, scaler_y_file) 