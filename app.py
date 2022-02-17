#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    val1 = request.form['IA1']
    val2 = request.form['IA2']
    val3 = request.form['IA3']
    val4 = request.form['AVG']
    prediction = model.predict([[val1,val2,val3,val4]])

    output = round(prediction[0])
    #render_template('index.html', prediction_text="Your Final Score may be")
    return render_template('index.html', prediction_text='Your Final Score may be around:{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)