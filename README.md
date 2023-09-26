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
Run
```console
# Build a local docker image
docker build -t <image_name> .
# Run the image
docker run -p 8080:8080 <image_name>
```

## Demo
Visit [demo](https://autom-coder-ml-methods-streamlit-main-g0wljb.streamlit.app/)