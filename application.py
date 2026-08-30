from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)
app = application


# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Prediction page
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    # When opening the prediction page
    if request.method == "GET":
        return render_template("home.html")

    try:
        # -----------------------------
        # Get data from HTML form
        # -----------------------------
        print("FORM DATA:")
        print(request.form)

        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        # -----------------------------
        # Convert input into DataFrame
        # -----------------------------
        pred_df = data.get_data_as_data_frame()

        print("PREDICTION DATA:")
        print(pred_df)

        # -----------------------------
        # Load model and preprocessor
        # -----------------------------
        predict_pipeline = PredictPipeline()

        # -----------------------------
        # Make prediction
        # -----------------------------
        results = predict_pipeline.predict(pred_df)

        print("PREDICTION RESULT:")
        print(results)

        # -----------------------------
        # Return prediction to webpage
        # -----------------------------
        return render_template(
            "home.html",
            results=results[0]
        )

    except Exception as e:

        # Print error in AWS logs
        print("========================================")
        print("PREDICTION ERROR:")
        print(repr(e))
        print("========================================")

        # Show actual error in browser while debugging
        return f"""
        <html>
            <body>
                <h1>Prediction Error</h1>
                <p>{str(e)}</p>
                <br>
                <a href="/predictdata">Go Back</a>
            </body>
        </html>
        """, 500


# Run application locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)