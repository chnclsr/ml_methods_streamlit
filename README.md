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
Derste 
docker run -p 8080:8080 mlimage bash ile çalıştırmak istemiştik, o metot ile farklı bilgisayarlarda denediğimde sorun olmadı. Aşağıdaki şekliyle doğrudan çalışacaktır.

Run
```console
# Build a local docker image
docker build -t mlimage .
# Run the image
docker run -p 8080:8080 mlimage
```

## Demo
Visit [demo](https://autom-coder-ml-methods-streamlit-main-g0wljb.streamlit.app/)
