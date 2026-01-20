# Butterfly Species Classifier

A Streamlit application that uses a Deep Learning model to classify butterfly species from uploaded images.

## Features

- **Image Upload**: Upload `.jpg`, `.jpeg`, or `.png` images of butterflies.
- **Real-time Prediction**: Instantly classifies the butterfly species using a pre-trained Keras model.
- **Confidence Score**: Displays the confidence level of the prediction.
- **Comparison**: Compares the predicted label with the actual label (derived from the filename).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd butterfly-classifier-app-main
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

2.  **Open in Browser**: The app will open automatically in your default browser (usually at `http://localhost:8501`).

3.  **Classify**: Upload an image to see the prediction.

## Files

- `app.py`: Main Streamlit application code.
- `Butterfly_classification.keras`: Pre-trained Keras model.
- `class_indices.pkl`: Dictionary mapping class indices to labels.
- `requirements.txt`: List of Python dependencies.
