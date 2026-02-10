from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))

# Compatibility shim: some DecisionTreeClassifier instances pickled
# with older scikit-learn versions include attributes newer versions
# do not have. Add a default `monotonic_cst` attribute to any
# deserialized DecisionTreeClassifier to avoid AttributeError at
# prediction time when using a newer scikit-learn.
def _ensure_monotonic_attr_on_trees(estimator):
    try:
        from sklearn.tree import DecisionTreeClassifier
    except Exception:
        DecisionTreeClassifier = None

    # If this is an ensemble with base estimators (e.g., RandomForest)
    estimators = getattr(estimator, 'estimators_', None)
    if estimators:
        for e in estimators:
            if DecisionTreeClassifier is None or isinstance(e, DecisionTreeClassifier):
                if not hasattr(e, 'monotonic_cst'):
                    setattr(e, 'monotonic_cst', None)

    # If the estimator itself is a DecisionTreeClassifier
    if DecisionTreeClassifier is None or isinstance(estimator, DecisionTreeClassifier):
        if not hasattr(estimator, 'monotonic_cst'):
            setattr(estimator, 'monotonic_cst', None)


# Apply shim to the loaded model
try:
    _ensure_monotonic_attr_on_trees(model)
except Exception:
    pass

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)


if __name__ == "__main__":
    app.run(debug=True)