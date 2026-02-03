import pickle

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm


print("Loading data...")
data_dict = pickle.load(open('./app/Python/Data&modeltraining/data.pickle', 'rb'))
print(f"Data keys: {data_dict.keys()}")

# Pad data with progress bar - FIXED TO PAD TO 84 FEATURES
print("\nPreparing data (padding sequences)...")
max_length = max(len(row) for row in data_dict['data'])
print(f"Max length found in data: {max_length}")

# Force padding to 84 features
TARGET_LENGTH = 84
print(f"Padding all samples to {TARGET_LENGTH} features...")

data = []
for row in tqdm(data_dict['data'], desc="Padding data", unit="sample"):
    if len(row) < TARGET_LENGTH:
        # Pad with zeros if shorter than 84
        data.append(row + [0] * (TARGET_LENGTH - len(row)))
    elif len(row) > TARGET_LENGTH:
        # Truncate if longer than 84 (shouldn't happen, but just in case)
        data.append(row[:TARGET_LENGTH])
    else:
        # Exactly 84 features
        data.append(row)

data = np.array(data)
labels = np.asarray(data_dict['labels'])

print(f"\nData shape: {data.shape}")
print(f"Number of features per sample: {data.shape[1]}")
print(f"Number of samples: {len(labels)}")
print(f"Unique classes: {len(set(labels))}")

print("\nSplitting data into train/test sets...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")

# Scale features for better SVM performance
print("\nScaling features...")
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("\nTraining SVM model...")
# ADDED probability=True to enable predict_proba
model = SVC(
    C=100, gamma=0.01, kernel='rbf', probability=True, class_weight='balanced'
)

model.fit(x_train_scaled, y_train)

print("\nMaking predictions on test set...")
y_predict = model.predict(x_test_scaled)

score = accuracy_score(y_predict, y_test)

print('\n' + '='*50)
print(f'✓ {score * 100:.2f}% of samples were classified correctly!')
print('='*50)

print("\nSaving model and scaler to ./app/Python/model_svm.p...")
f = open('./app/Python/model_svm.p', 'wb')
pickle.dump({'model': model, 'scaler': scaler}, f)
f.close()

print("Done! Model and scaler saved successfully.")
print(f"Model expects {data.shape[1]} features as input.")