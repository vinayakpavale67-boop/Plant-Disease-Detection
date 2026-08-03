from flask import Flask, render_template, request
import os
from predict import predict_disease

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image selected"

    file = request.files["image"]

    if file.filename == "":
        return "Please choose an image"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    prediction, confidence = predict_disease(filepath)

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        image_path=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)