import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
 #flask app
flask_app=Flask(__name__)

 #load pickle model
model=pickle.load(open("model.pkl","rb"))
#we will use app to define homepage
#the below code would direct to the homepage

@flask_app.route("/")
def Home():
  return render_template("index.html")

@flask_app.route("/predict",methods=["POST"])
def predict():
 float_features=[float(x) for x in request.form.values()]
 features =[np.array(float_features)]
 return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)
