from flask import Flask, render_template, request
from predict import predict_image

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    score = None

    if request.method == "POST":
        file = request.files["file"]
        path = "uploaded.jpg"
        file.save(path)

        result, score = predict_image(path)

    return render_template("index.html", result=result, score=score)

if __name__ == "__main__":
    app.run(debug=True)

