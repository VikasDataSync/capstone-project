from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/VikasDataSync/capstone-project.mlflow')
# dagshub.init(repo_owner='VikasDataSync', repo_name='capstone-project', mlflow=True)
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "VikasDataSync"
repo_name = "capstone-project"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# --------------------------------------------------------------------------------


# Initialize Flask app
app = Flask(__name__)

# from prometheus_client import CollectorRegistry

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup
model_name = "my_model"
ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
LOCAL_VECTORIZER_PATH = ROOT_DIR / "models" / "vectorizer.pkl"


def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    for stage in ["Production", "Staging", "None"]:
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        if latest_version:
            return latest_version[0].version

    all_versions = client.search_model_versions(f"name='{model_name}'")
    if all_versions:
        return max(all_versions, key=lambda version: int(version.version)).version

    return None


def load_prediction_model(model_name):
    model_version = get_latest_model_version(model_name)
    if model_version is not None:
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"Fetching model from: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)

    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No registered model versions found for '{model_name}', and local model was not found at '{LOCAL_MODEL_PATH}'."
        )

    print(
        f"No registered versions found for '{model_name}'. Loading local model from '{LOCAL_MODEL_PATH}'."
    )
    with open(LOCAL_MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


model = load_prediction_model(model_name)
with open(LOCAL_VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    # Clean text
    text = normalize_text(text)
    # Convert to features
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict
    result = model.predict(features_df)
    prediction = result[0]

    # Increment prediction count metric
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    # Measure latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker
