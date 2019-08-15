# Load libraries

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics


sample = pd.read_csv("/Users/sireeshapandeti/Documents/ArtificialIntelligence/Student_Admission_Predict.csv")
print(sample.head())


#split dataset in features and target variable
feature_cols = ['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating','SOP','LOR ','CGPA']
X = sample[feature_cols] # Features

y = sample.Research # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols, class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('sample.png')
Image(graph.create_png())















