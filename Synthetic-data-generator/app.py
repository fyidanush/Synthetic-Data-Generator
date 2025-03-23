from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
from data_ingestion import DataIngestion
from data_generation import train_gan
from data_validation import compare_distributions

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    # Get the uploaded file
    file = request.files["file"]
    if file:
        # Load and preprocess the data
        data_ingestion = DataIngestion(file)
        data = data_ingestion.load_data()

        # Ensure the data is numeric
        if not all(data.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
            return jsonify({"error": "Non-numeric data detected. Please upload a numeric dataset."}), 400

        # Convert data to PyTorch tensor
        real_data_tensor = torch.tensor(data.values, dtype=torch.float32)

        # Train GAN and generate synthetic data
        generator = train_gan(real_data_tensor)
        noise = torch.randn(real_data_tensor.shape[0], real_data_tensor.shape[1])
        synthetic_data_tensor = generator(noise)
        synthetic_data = pd.DataFrame(synthetic_data_tensor.detach().numpy(), columns=data.columns)

        # Validate the synthetic data
        validation_results = compare_distributions(data, synthetic_data)

        # Return results as JSON
        return jsonify({
            "synthetic_data": synthetic_data.to_dict(orient="records"),
            "validation_results": validation_results
        })
    else:
        return jsonify({"error": "No file uploaded"}), 400

if __name__ == "__main__":
    app.run(debug=True)