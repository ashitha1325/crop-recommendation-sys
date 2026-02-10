import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
crop = pd.read_csv('Crop_recommendation.csv')

print("Dataset shape:", crop.shape)
print("Columns:", crop.columns.tolist())
print("\nLabel value counts:")
print(crop['label'].value_counts())

# Map labels to integers
crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}

crop['label'] = crop['label'].map(crop_dict)

# Prepare features and target
X = crop.drop('label', axis=1)
y = crop['label']

print("\nFeature columns:", X.columns.tolist())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Fit MinMaxScaler on training data
mx = MinMaxScaler()
X_train_mx = mx.fit_transform(X_train)
X_test_mx = mx.transform(X_test)

# Fit StandardScaler on MinMax-scaled training data
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_mx)
X_test_sc = sc.transform(X_test_mx)

# Train RandomForest model
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train_sc, y_train)

# Evaluate
y_pred = model.predict(X_test_sc)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")

# Save model and scalers - DO NOT call fit_transform again after this!
print("\nSaving model and scalers...")
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

print("Saved: model.pkl, minmaxscaler.pkl, standscaler.pkl")

# Verify by loading and testing
print("\n--- Verification ---")
model_loaded = pickle.load(open('model.pkl', 'rb'))
mx_loaded = pickle.load(open('minmaxscaler.pkl', 'rb'))
sc_loaded = pickle.load(open('standscaler.pkl', 'rb'))

# Test with a sample from the dataset (first rice sample)
rice_sample = crop[crop['label'] == 1].iloc[0]
test_features = np.array([[rice_sample['N'], rice_sample['P'], rice_sample['K'], 
                           rice_sample['temperature'], rice_sample['humidity'], 
                           rice_sample['ph'], rice_sample['rainfall']]])

print(f"Test input (Rice sample): N={rice_sample['N']}, P={rice_sample['P']}, K={rice_sample['K']}")
print(f"  temp={rice_sample['temperature']:.2f}, humidity={rice_sample['humidity']:.2f}, ph={rice_sample['ph']:.2f}, rainfall={rice_sample['rainfall']:.2f}")

# Transform using loaded scalers (NOT fit_transform!)
test_mx = mx_loaded.transform(test_features)
test_sc = sc_loaded.transform(test_mx)
prediction = model_loaded.predict(test_sc)

crop_names = {v: k for k, v in crop_dict.items()}
print(f"Predicted: {prediction[0]} ({crop_names.get(prediction[0], 'Unknown')})")
print(f"Expected: 1 (rice)")

# Test with a few more samples
print("\n--- Additional Tests ---")
for label_num, label_name in [(2, 'maize'), (8, 'apple'), (22, 'coffee')]:
    sample = crop[crop['label'] == label_num].iloc[0]
    features = np.array([[sample['N'], sample['P'], sample['K'], 
                          sample['temperature'], sample['humidity'], 
                          sample['ph'], sample['rainfall']]])
    features_mx = mx_loaded.transform(features)
    features_sc = sc_loaded.transform(features_mx)
    pred = model_loaded.predict(features_sc)[0]
    print(f"{label_name}: predicted={pred} ({crop_names.get(pred, '?')}), expected={label_num}")

print("\nTraining complete!")
