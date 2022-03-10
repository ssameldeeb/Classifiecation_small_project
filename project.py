import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

now = datetime.datetime.now()

data = pd.read_csv("spotify_top50_2021.csv")
print(data.shape)
print(data.isnull().sum())

data = data.dropna()
print(data.isnull().sum().sum())
print(data.dtypes)
print(data.head())
print(data.columns.values)

data = data.drop(["track_id","id"], axis=1)
print(data.shape[1])

ch = ["track_name","artist_name"]
for x in ch:
    La = LabelEncoder()
    data[x] = La.fit_transform(data[x])
print(data.head())

plt.figure(figsize=(14,7))
sns.heatmap(data.corr(), annot=True)
plt.show()

now1 = datetime.datetime.now()
print(now1 - now)

x = data.drop("mode", axis=1)
y = data["mode"]
ss = StandardScaler()
x = ss.fit_transform(x)
print(x.shape)
print(y.shape)

DT = DecisionTreeClassifier()
DT.fit(x, y)

print(DT.score(x,y))
print(DT.feature_importances_)

inp = [0.2 , 0.50, 0.09, 0.3, 0.4, 0.1, 0.5, 0.4, 0.66, 0.7, 0.044, 0.4, 0.1, 0.5, 0.4]
print(len(inp))

print(DT.predict([inp]))