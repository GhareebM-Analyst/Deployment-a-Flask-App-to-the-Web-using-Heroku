import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
model=pickle.load(open('model_final.pkl','rb'))
scalar=pickle.load(open('stand_scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=round(np.exp(model.predict(final_input)[0]),2)
    return render_template("home.html",prediction_text="The Estimated is is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
