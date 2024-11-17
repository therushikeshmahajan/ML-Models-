import numpy as np
import pandas as pd
import os as os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def find_file(filename = 'Data Source', file_ext = ".csv", search_path = "/Users/rushikeshmahajan/Desktop/Documents/"):
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file == 'Data Source'+'.csv':
                return os.path.join(root,file)
    return print("No files found")

def load_data(filepath):
    if filepath.endswith(".csv"):
        print("File Found")
        return pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format.")

file_path = find_file()
if file_path:
    print(f'file found: {file_path}')
    data = load_data(file_path)
    print(f'data loaded successfully.')
else:
    print("File not found")
    exit 

'''print(data)'''

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data['class'] = data['Fraud'].apply(lambda x : 1 if x == "Yes" else 0)

X = data[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']].copy()
y = data['Fraud'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print(f"y_pred: {y_pred}")  # Print y_pred values next to the variable name
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))







