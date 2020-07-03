## Text classification using scikit-learn, PyTorch, and TensorFlow
Build text classifiers using 3 most popular machine learning or deep learning libraries - Scikit-learn, PyTorch, TensorFlow

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

### Installation <a name="installation"></a>
You can download anaconda individual edition from https://www.anaconda.com/products/individual, which contains all the useful libraries used by data scientists. 
Another option is to intall the following package use pip package manager. 
python                    3.8.3
tensorflow                2.2.0 
torch                     1.5.1 
jupyterlab                2.1.4 
pandas                    1.0.4

### Project Motivation<a name="motivation"></a>
Text classification has been widely used in real world business processes like email spam detection, support ticket classification, or content recommendation based on text topics.
I would like to build multi-class text classfier using the 3 most popular open source machine learning or deep learning libraries: scikit-learn, PyTorch, and TensorFlow. I am interested in seeing how they perform comparing to each other. 

### File Descriptions <a name="files"></a>
1. gather_explore_data.ipynb: Gathers sample data used for this project and explore how the data look like
2. feature_extraction.ipynb: Transforms texts or words into numerical vector representation in order to feed into models for training
3. util.py: The help functions for feature extraction
4. model_scikit_learn.ipynb: Build and train text classifiers using Scikit Learn
5. model_pytorch.ipynb: Build and train text classification using PyTorch
6. model_tensorflow_tfidf.ipynb: Build and train text classification using TensorFlow, and encoding input texts using TF-IDF algorithm
7. model_tensorflow.ipynb: Build and train tect classification using TensorFlow, and encode imput text using padded sequences. Also apply word embedding.

### Results<a name="results"></a>
The result can be found at the post available at https://medium.com/@donglinchen/text-classification-using-scikit-learn-pytorch-and-tensorflow-a3350808f9f7

### Licensing, Authors, Acknowledgements<a name="licensing"></a>
Sample data are available at: https://www.kaggle.com/yufengdev/bbc-fulltext-and-category