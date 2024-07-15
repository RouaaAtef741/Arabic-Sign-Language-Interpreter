import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

print("Initial data shapes:")
for i, sample in enumerate(data):
    print(f"Sample {i}: {len(sample)} landmarks")

landmarks_per_hand = len(data[0])  

print(f"Landmarks per hand: {landmarks_per_hand}")

processed_data = []
processed_labels = []

for i, (sample, label) in enumerate(zip(data, labels)):
    sample_length = len(sample)
    if sample_length == landmarks_per_hand:
        processed_data.append(sample)
        processed_labels.append(label)
    else:
        print(f"Warning: Sample {i} has an unexpected number of landmarks ({sample_length}). Skipping.")
        continue

processed_data = np.array(processed_data)
processed_labels = np.array(processed_labels)

print(f"Processed data shape: {processed_data.shape}")
print(f"Processed labels shape: {processed_labels.shape}")

x_train, x_test, y_train, y_test = train_test_split(processed_data, processed_labels, test_size=0.2, shuffle=True, stratify=processed_labels)

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)
print(f"Flattened x_train shape: {x_train_flat.shape}, Flattened x_test shape: {x_test_flat.shape}")

model = RandomForestClassifier()
model.fit(x_train_flat, y_train)

y_predict = model.predict(x_test_flat)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
