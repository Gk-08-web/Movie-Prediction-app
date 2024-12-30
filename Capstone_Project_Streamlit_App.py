import pandas as pd
import numpy as np

file_path = (open("movie_metadata_renamed.csv"))
data = pd.read_csv(file_path)

data.head()   #Load and Examine the dataset

print ("Dataset Shape:",
       data.shape)

print("Data Info:")
print(data.info())

print("Summary Statistics:")
print(data.describe()) #understand dataset structure

#Performing Exploratory Data Analysis (EDA)

# Identify missing values
missing_values = data.isnull().sum()
print("Missing Values in Each Column:\n", missing_values)

# Handle missing values
# Fill missing numeric values with the median
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Fill missing categorical values with the mode
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Verify missing values are handled
print("\nMissing Values After Handling:\n", data.isnull().sum())

# Distribution of IMDb scores
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data['imdb_score'], bins=20, kde=True)
plt.title("Distribution of IMDb Scores")
plt.xlabel("IMDb Score")
plt.ylabel("Frequency")
plt.show()

# Scatterplot: Budget vs IMDb Score
sns.scatterplot(x='budget', y='imdb_score', data=data, alpha=0.7)
plt.title("Budget vs IMDb Score")
plt.xlabel("Budget")
plt.ylabel("IMDb Score")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Select Numeric Columns Only
numeric_data = data.select_dtypes(include=['int64', 'float64'])


if numeric_data.shape[1] > 1:  
    correlation_matrix = numeric_data.corr()

    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("Not enough numeric data to generate a correlation heatmap.")

print(data.columns)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# First, let's split the genres and get individual genre counts
def split_genres(genre_string):
    return str(genre_string).split('|')

# Create a list of all genres
all_genres = []
for genres in data['genres']:
    all_genres.extend(split_genres(genres))
genre_counts = pd.Series(all_genres).value_counts()

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Top subplot: Box plot of IMDb scores by genre
plt.subplot(211)
genre_imdb = []
genre_names = []

for genre in genre_counts.index[:10]:  # Top 10 genres
    # Get movies containing this genre
    mask = data['genres'].str.contains(genre, na=False)
    scores = data[mask]['imdb_score']
    genre_imdb.extend(scores)
    genre_names.extend([genre] * len(scores))

genre_data = pd.DataFrame({
    'Genre': genre_names,
    'IMDb Score': genre_imdb
})

sns.boxplot(x='Genre', y='IMDb Score', data=genre_data, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_title('IMDb Scores Distribution by Top 10 Genres', pad=20)

# Bottom subplot: Genre frequency bar plot
plt.subplot(212)
genre_counts[:10].plot(kind='bar')
plt.title('Top 10 Most Common Genres', pad=20)
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Print average IMDb score by genre
print("\nAverage IMDb Score by Genre:")
for genre in genre_counts.index[:10]:
    mask = data['genres'].str.contains(genre, na=False)
    avg_score = data[mask]['imdb_score'].mean()
    print(f"{genre}: {avg_score:.2f}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create figure with larger size for better readability
plt.figure(figsize=(15, 8))

# Create boxplot with enhanced styling
sns.boxplot(data=data, 
           x='imdb_score',
           y='content_rating',
           palette='viridis',
           orient='h')  # Horizontal orientation for better label readability

# Customize the plot
plt.title('IMDb Scores Distribution by Content Rating', fontsize=14, pad=20)
plt.xlabel('IMDb Score', fontsize=12)
plt.ylabel('Content Rating', fontsize=12)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.7, axis='x')

# Tight layout to prevent label cutoff
plt.tight_layout()

# Show plot
plt.show()

# Print summary statistics
print("\nSummary Statistics by Content Rating:")
summary_stats = data.groupby('content_rating')['imdb_score'].describe()
print(summary_stats)

# Genre frequency and IMDb scores by genre
def split_genres(genre_string):
    return str(genre_string).split('|')

all_genres = []
for genres in data['genres']:
    all_genres.extend(split_genres(genres))

genre_counts = pd.Series(all_genres).value_counts()
print("Genre Frequency:\n", genre_counts)


def categorize_imdb(score):
    if score >= 6:
        return 'Hit'
    elif score >= 3:
        return 'Average'
    else:
        return 'Flop'

# Categorize IMDb scores into classes
data['Category'] = data['imdb_score'].apply(categorize_imdb)

from sklearn.preprocessing import LabelEncoder

# Encode the target variable
label_encoder = LabelEncoder()
data['Category_encoded'] = label_encoder.fit_transform(data['Category'])

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset
file_path = "C:/Users/gargi/MoviePrediction/movie_metadata.csv.csv"
data = pd.read_csv(file_path)

# Handle missing values (as shown in your previous code)
# Fill missing numeric values with the median
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Define numeric_features
numeric_features = numeric_cols.tolist()  # Convert the Index object to a list

# Encode the target variable
label_encoder = LabelEncoder()
data['Category'] = data['imdb_score'].apply(lambda score: 'Hit' if score >= 6 else ('Average' if score >= 3 else 'Flop'))
data['Category_encoded'] = label_encoder.fit_transform(data['Category'])

# Define features and target variable
X = data[numeric_features]  # Features
y = data['Category_encoded']  # Encoded target variable

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Unique Target Labels (y):", y.unique())

# Categorize IMDb Scores
def categorize_imdb(score):
    if score >= 6:
        return 'Hit'
    elif score >= 3:
        return 'Average'
    else:
        return 'Flop'

data['Category'] = data['imdb_score'].apply(categorize_imdb)

# Check the distribution of categories
print(data['Category'].value_counts())

#Model Selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Encode target variable
label_encoder = LabelEncoder()
data['Category_encoded'] = label_encoder.fit_transform(data['Category'])

# Prepare the feature set and target
X = data[numeric_features]
y = data['Category_encoded']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Model Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': numeric_features, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Display the top important features
print("\nFeature Importances:\n", importance_df)

#Evaluate Other Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

# Compare models using cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define a function to evaluate models
def evaluate_model(model, X_test, y_test, y_pred):
    print(f"\nModel: {model}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Evaluate Random Forest
rf_y_pred = rf.predict(X_test)
evaluate_model("Random Forest", X_test, y_test, rf_y_pred)

# Evaluate other models (use y_pred from earlier evaluations for each model)
# Example for Logistic Regression (replace with your trained model objects)
# lr_y_pred = logistic_regression.predict(X_test)
# evaluate_model("Logistic Regression", X_test, y_test, lr_y_pred)


#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Use the best model
best_rf = grid_search.best_estimator_

import joblib

# Save the model
joblib.dump(rf, 'random_forest_model.pkl')

# To load the model in the future
# loaded_model = joblib.load('random_forest_model.pkl')

# Load the model
model = joblib.load('random_forest_model.pkl')

# Example input data
new_data = [[120, 2000, 500, 300, 100, 50, 20, 10000, 5000, 2000, 300000,50000,10,40,3000,60000]]

# Make prediction
prediction = model.predict(new_data)
print("Prediction:", label_encoder.inverse_transform(prediction))  # Decode the label

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, display_labels=label_encoder.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize the labels for ROC (if needed)
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])  # Modify based on number of classes
rf_probs = best_rf.predict_proba(X_test)

# Compute ROC for one class (e.g., class 1)
fpr, tpr, _ = roc_curve(y_test_binarized[:, 1], rf_probs[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

# Feature Importance Bar Chart
importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False, figsize=(10, 6), color='skyblue')
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

import joblib

# Assuming 'rf' is your trained Random Forest model
joblib.dump(rf, 'random_forest_model.pkl')

# Save the label encoder if you used one
joblib.dump(label_encoder, 'label_encoder.pkl')

