%pip install shap xgboost
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
np.random.seed(123)
import shap
shap.initjs()
import warnings
warnings.filterwarnings ('ignore')

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names) 
df['target'] = data.target
df.head()

#TRAINING XG BOOST
y=df['target'].to_frame() # define Y 
X=df[df.columns.difference(['target'])] # define X 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # create train and test 

model = XGBClassifier(random_state=42,gpu_id=0) # build classifier Gradient Boosted decision trees
model.fit(X_train,y_train.values.ravel())

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#SHAP EXPLAINER
explainer = shap.TreeExplainer(model)

#Storing Shap expected value
shap_values = explainer.shap_values(X)
expected_value = explainer.expected_value

#Explaining XGBoost AI using Shap explainer
shap.summary_plot(shap_values, X,title="SHAP summary plot")

shap.summary_plot(shap_values, X,plot_type="bar", feature_names=data.feature_names , show=True)
