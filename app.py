from model import StackingEstimator
import joblib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for


#app = Flask(__name__, template_folder = 'template')
app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #
    #For rendering results on HTML
    #
    int_features = [int(x) for x in request.form.values()]
 
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)
    
    def final_countdown(prediction):
        if prediction == 1.0:
            return ("compre")
        else:
            return ("no compre")
    deneme1 = final_countdown(round(prediction[0],2))
    
    #output = round(prediction[0],2)
    return render_template('index.html', prediction_text = 'Para los valores dados, es más probable que este cliente {} el producto del banco'.format(deneme1))
    #return render_template('index.html', prediction_text = 'Para los valores dados, es más probable que este cliente {} sea el producto deseado del banco'.format(deneme1))

if __name__ == "__main__":
    app.run(port=5000, debug=True)

