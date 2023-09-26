# Interactive Web App with Streamlit and Scikit-learn
Explore different datasets and classifier. Streamlit lets you create apps for your machine learning projects with simple
Python scripts. See official [streamlit website](https://www.streamlit.io/) for more info.

## Installation
You need these dependencies:
```console
pip install streamlit
pip install scikit-learn
pip install matplotlib
```

## Usage - Local
Run
```console
streamlit run main.py
```

## Usage - Docker

```console
docker build -t streamlit_ml_classifier .
docker run --network host --name container_streamlit -p 8501:8501 -it streamlit_ml_classifier bash
```