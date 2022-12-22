import pandas as pd
import numpy as np
import lime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer
from lime.lime_tabular import LimeTabularExplainer
from lime import submodular_pick
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Loading the data to work with
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names) 
df['target'] = data.target
df.head()

X = data["data"]
y = data["target"]
features = data.feature_names

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state = 42)

model = XGBClassifier(n_estimators = 300,random_state = 123)
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy: %.2f%%" %(accuracy*100))

predict_fn = lambda x: model.predict_proba(x)

np.random.seed(123)
explainer = LimeTabularExplainer(df[features].astype(int).values,mode = "classification", class_names = 
                                 ["Positive","Negative"],training_labels= df['target'],feature_names = features)

i = 5
exp = explainer.explain_instance(df.loc[i,features].astype(int).values,predict_fn,num_features = 5)
exp.show_in_notebook(show_table = True)

figure = exp.as_pyplot_figure(label = exp.available_labels()[0])
