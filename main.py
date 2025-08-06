import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load CSV
df = pd.read_csv('earthquake_catalog.csv', parse_dates=['time'])

# Sort and compute rolling features BEFORE dropna
df = df.sort_values('time')
df['count_7d'] = df['time'].rolling(7).count().shift(1)
df['avg_mag_7d'] = df['magnitude'].rolling(7).mean().shift(1)

# Create binary target
df['target'] = (df['magnitude'] >= 4.0).astype(int)

# Drop missing rows AFTER rolling
df = df.dropna(subset=['count_7d', 'avg_mag_7d', 'target'])

# Features and target
x = df[['count_7d', 'avg_mag_7d']]
y = df['target']

# Check if enough data
if len(df) < 5:
    print("Not enough data to train/test. Need more samples.")
    exit()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)


# Train
clf = RandomForestClassifier(n_estimators=50, random_state=0)
clf.fit(x_train, y_train)
print("Model trained!")

# Predict
y_pred = clf.predict(x_test)

# Output
print("Predicted labels:", y_pred)
print("Actual labels:   ", y_test.values)

# Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
