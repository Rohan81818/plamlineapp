import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load labeled data
data = pd.read_csv('length-orientation-centroidx-centroidy-label.csv')

# Fill any missing orientation values with median
data['orientation'] = data['orientation'].fillna(data['orientation'].median())

# Features and target
X = data[['length', 'orientation', 'centroid_x', 'centroid_y']]
y = data['label']

# Label encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split without stratify due to small sample sizes
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Prepare label indexes for classification report
all_labels = list(range(len(le.classes_)))

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, labels=all_labels, target_names=le.classes_))

# Save model and label encoder for later use
joblib.dump(clf, 'palm_line_classifier.joblib')
joblib.dump(le, 'label_encoder.joblib')
