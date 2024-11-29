import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')

sns.set_style("whitegrid", {'axes.grid': False})

################################################################################
#                                                                              #
#                            Analyze Data                                      #
#                                                                              #
################################################################################
def explore_data(df):
    print("Number of Instances and Attributes:", df.shape)
    print('\n')
    print('Dataset columns:', df.columns)
    print('\n')
    print('Data types of each column: ', df.info())

################################################################################
#                                                                              #
#                      Checking for Duplicates                                 #
#                                                                              #
################################################################################
def checking_removing_duplicates(df):
    count_dups = df.duplicated().sum()
    print("Number of Duplicates: ", count_dups)
    if count_dups >= 1:
        df.drop_duplicates(inplace=True)
        print('Duplicate values removed!')
    else:
        print('No Duplicate values')

################################################################################
#                                                                              #
#                Split Data to Training and Validation set                     #
#                                                                              #
################################################################################
def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

################################################################################
#                                                                              #
#                           Train Model                                        #
#                                                                              #
################################################################################
def fit_model(X_train, y_train, models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    return names, results

################################################################################
#                                                                              #
#                          Save Trained Model                                  #
#                                                                              #
################################################################################
def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))

################################################################################
#                                                                              #
#                          Performance Measure                                 #
#                                                                              #
################################################################################
def classification_metrics(model, X_train, y_train, X_test, y_test, y_pred, conf_matrix):
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    print(classification_report(y_test, y_pred))

# Load Dataset
df = pd.read_csv('Crop_recommendation.csv')

# Remove Outliers - apply only to numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Split Data to Training and Validation set
target = 'label'
X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)

# Train model
pipeline = make_pipeline(StandardScaler(), GaussianNB())
model = pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Pass necessary arguments to the function
classification_metrics(pipeline, X_train, y_train, X_test, y_test, y_pred, conf_matrix)

# Save model
save_model(model, 'model.pkl')
