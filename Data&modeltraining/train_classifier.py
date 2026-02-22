import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('data.pickle', 'rb'))
max_length = max(len(row) for row in data_dict['data'])
data = np.array([row + [0] * (max_length - len(row)) for row in data_dict['data']])
labels = np.asarray(data_dict['labels'])
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier(n_estimators=70,           # More trees = better accuracy (increased from 100)
    max_depth=None,             # Allow trees to grow fully for complex patterns
    min_samples_split=2,        # Minimum samples to split a node
    min_samples_leaf=1,         # Minimum samples at leaf node
    max_features='sqrt',        # Number of features to consider for best split
    bootstrap=True,             # Use bootstrap samples
    class_weight='balanced',    # Handle class imbalance automatically
    n_jobs=-1,                  # Use all CPU cores for faster training
    random_state=42,
    verbose=2,
    warm_start=False,
    oob_score=True )

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('./app/Python/model2.p', 'wb')
pickle.dump({'model': model}, f)
f.close()