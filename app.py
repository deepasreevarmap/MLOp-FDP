from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

species_map = {0:"Setosa",1:"Versicolor",2:"Virginica"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["sepal_length"]),
        float(request.form["sepal_width"]),
        float(request.form["petal_length"]),
        float(request.form["petal_width"])
    ]

    prediction = model.predict([features])[0]
    result = species_map[prediction]

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
