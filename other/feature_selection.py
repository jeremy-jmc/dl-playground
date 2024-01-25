
# TODO: https://ai.gopubby.com/ml-tutorial-13-feature-selection-methods-and-importance-93221a37abaf
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.datasets import load_breast_cancer, fetch_covtype, fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

SEED = 42

# Load the dataset
dataset = fetch_openml(name='higgs', version=2)
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
df = df.dropna(how='any')
X, y = df.drop('target', axis=1), df['target']
y = y.astype(int)

# # Min-max normalization
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X = X + abs(X.min())    # no negative values


# Define a function to evaluate the performance of each feature selection technique
def evaluate_performance(X_train, X_test, y_train, y_test, feature_names, technique_name):
    # Import the logistic regression model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    # Train the model on the selected features
    model = XGBClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Compute the accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred)
    # Compute the ROC curve and AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="AUC = {:.2f}".format(auc_score))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for {}".format(technique_name))
    plt.legend()
    plt.show()
    # Print the performance metrics
    print("Performance metrics for {}:".format(technique_name))
    print("Accuracy = {:.2f}".format(accuracy))
    print("Precision = {:.2f}".format(precision))
    print("Recall = {:.2f}".format(recall))
    print("F1-score = {:.2f}".format(f1))
    # Print the selected features
    print(f"{len(feature_names)} selected features for {technique_name}:")
    print(feature_names)

# Assume X and y are your feature matrix and target variable, respectively
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f'X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}')

# 0. No Feature Selection
evaluate_performance(X_train, X_test, y_train, y_test, X.columns, "No Feature Selection")

# 1. Variance Threshold
from sklearn.feature_selection import VarianceThreshold

def variance_threshold_selection(X_train, X_test, threshold=0.0):
    selector = VarianceThreshold(threshold=threshold)
    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)
    feature_names = [f for f, s in zip(X.columns, selector.get_support()) if s]
    return X_train_selected, X_test_selected, feature_names

X_train_var, X_test_var, feature_names_var = variance_threshold_selection(X_train, X_test, threshold=0.1)
evaluate_performance(X_train_var, X_test_var, y_train, y_test, feature_names_var, "Variance Threshold")

# 2. Correlation Coefficient
def correlation_coefficient_selection(X_train, X_test, threshold=0.8):
    corr_matrix = X_train.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_train_selected = X_train.drop(to_drop, axis=1)
    X_test_selected = X_test.drop(to_drop, axis=1)
    feature_names = list(X_train_selected.columns)
    return X_train_selected, X_test_selected, feature_names

X_train_corr, X_test_corr, feature_names_corr = correlation_coefficient_selection(X_train, X_test, threshold=0.8)
evaluate_performance(X_train_corr, X_test_corr, y_train, y_test, feature_names_corr, "Correlation Coefficient")

# 3. Chi-Square Test
from sklearn.feature_selection import chi2, SelectKBest

def chi_square_selection(X_train, X_test, y_train, k=5):
    selector = SelectKBest(score_func=chi2, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    feature_names = [f for f, s in zip(X.columns, selector.get_support()) if s]
    return X_train_selected, X_test_selected, feature_names

X_train_chi, X_test_chi, feature_names_chi = chi_square_selection(X_train, X_test, y_train, k=20)
evaluate_performance(X_train_chi, X_test_chi, y_train, y_test, feature_names_chi, "Chi-Square Test")

# Continue with similar code for ANOVA Test, Mutual Information, PCA, RFE, Lasso Regression, and Random Forest.
# Make sure to adapt the function names and parameters accordingly