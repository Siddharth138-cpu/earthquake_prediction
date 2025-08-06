import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as Plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# Load data
df = pd.read_csv('earthquake_catalog.csv')
print("Available columns:", df.columns.tolist())  # Debug: see actual column names

# Create a fake feature (e.g., using magnitude itself)
df = df.dropna(subset=['magnitude'])

df['count_7d'] = df['magnitude'].rolling(window=7, min_periods=1).count()
df['avg_mag_7d'] = df['magnitude'].rolling(window=7, min_periods=1).mean()
df['target'] = (df['magnitude'] > 4.0).astype(int)

# Drop missing values in new columns
df = df.dropna(subset=['count_7d', 'avg_mag_7d', 'target'])

# Check sample size
if len(df) < 10:
    print("Not enough data to train/test. Need more samples.")
    exit()

# Features and labels
X = df[['count_7d', 'avg_mag_7d']]
y = df['target']

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model training
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model trained!")
print("Predicted:", y_pred.tolist())
print("Actual:   ", y_test.tolist())
print(f"Accuracy: {accuracy:.2f}")
'''
# Train the model
from sklearn.tree import DecisionTreeClassifier

# Define the model
clf = DecisionTreeClassifier()

# Now train it

clf.fit(x_train, y_train)

# Predict
y_pred = clf.predict(x_test)


# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=Plt.cm.Blues)

# Add title and show the plot
Plt.title("Confusion Matrix")
Plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt

# Just take the first N samples that both arrays have
N = min(len(y_test), len(y_pred))
x_axis = np.arange(N)

Plt.bar(x_axis - 0.2, y_test[:N], 0.4, label='Actual')
Plt.bar(x_axis + 0.2, y_pred[:N], 0.4, label='Predicted')

Plt.xlabel('Sample Index')
Plt.ylabel('Class')
Plt.title('Actual vs Predicted Comparison')
Plt.legend()
Plt.show()
