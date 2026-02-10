import pickle
import numpy as np

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Add compatibility shim
def _ensure_monotonic_attr_on_trees(estimator):
    try:
        from sklearn.tree import DecisionTreeClassifier
    except Exception:
        DecisionTreeClassifier = None
    estimators = getattr(estimator, 'estimators_', None)
    if estimators:
        for e in estimators:
            if DecisionTreeClassifier is None or isinstance(e, DecisionTreeClassifier):
                if not hasattr(e, 'monotonic_cst'):
                    setattr(e, 'monotonic_cst', None)
    if DecisionTreeClassifier is None or isinstance(estimator, DecisionTreeClassifier):
        if not hasattr(estimator, 'monotonic_cst'):
            setattr(estimator, 'monotonic_cst', None)

_ensure_monotonic_attr_on_trees(model)

# Test different inputs
test_cases = [
    [120, 60, 60, 30, 55, 7.2, 90],
    [50, 20, 30, 25, 60, 6.5, 100],
    [80, 50, 50, 28, 70, 7.0, 120],
    [90, 40, 45, 32, 50, 6.8, 80],
]

crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
             8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
             14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

print("Testing different inputs:")
print("=" * 80)
for feature_list in test_cases:
    single_pred = np.array(feature_list).reshape(1, -1)
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)
    
    print(f"Input: {feature_list}")
    print(f"  Raw prediction: {prediction[0]}")
    print(f"  Prediction type: {type(prediction[0])}")
    
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        print(f"  Crop: {crop}")
    else:
        print(f"  Crop: NOT IN DICT (prediction value: {prediction[0]})")
    print()
