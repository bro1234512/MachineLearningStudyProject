import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

np.random.seed(123)


# Paths to the data sets

data_path = "data.csv"

# Load data
data = pd.read_csv(data_path)

result = pd.DataFrame()
result['diagnosis'] = data.iloc[:, 1]
data = data.iloc[:, 2:-1]
label_encoder = LabelEncoder()
data.iloc[:, 0] = label_encoder.fit_transform(data.iloc[:, 0]).astype('float64')
x_train, x_test, y_train, y_test = train_test_split(data.values, result.values, test_size=0.2)
svc = SVC()
svc.fit(x_train, y_train)
prediction = svc.predict(x_test)
cm = confusion_matrix(y_test, prediction)
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]

accuracy = sum / x_test.shape[0]
print("Withput FS:" + accuracy)