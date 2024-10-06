# Deciphering Ancient Tamil Epigraphy: A Deep Learning Approach for Vatteluttu Script Recognition

Overview

This project implements a deep learning approach to recognize the Vatteluttu script used in ancient Tamil epigraphy. By utilizing convolutional neural networks (CNN) and a Siamese network architecture, this model aims to classify and recognize ancient Tamil characters from images effectively. The project is part of the ongoing research on the digitization of ancient texts, focusing on enhancing readability and accessibility.

To get started, clone the repository and install the required dependencies.

Required Dependencies 
- Python 3.x
- TensorFlow
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- scikit-learn

Data Preparation
Dataset link: https://www.kaggle.com/datasets/siddharthadevanv/8th-century-tamil-inscriptions 
The project utilizes images of ancient Tamil inscriptions, both ancient categorized and augmented as well as the modern images. 

Custom Image Preprocessing
The images undergo a series of preprocessing steps, including resizing, Gaussian smoothing, adaptive thresholding, and normalization. This enhances the model's ability to learn effectively from the data.

Model Architecture

The Siamese network architecture comprises the following layers:

Convolutional Layers: Extract features from images.
MaxPooling Layers: Reduce spatial dimensions.
Dropout Layers: Prevent overfitting.
LSTM Layers: Handle the sequential nature of data.
The model takes two inputs (ancient and modern characters) and learns to differentiate between them, ultimately classifying them into 28 distinct classes.

Training

The model is trained with early stopping to prevent overfitting. The training process includes:

- 20 epochs with early stopping (patience = 2)
- Batch size of 32
- Monitoring validation loss for optimal performance

Usage
To train the model, execute the cells in model.ipynb. Ensure that the data paths are correctly set and adjust hyperparameters as necessary.


Results

Upon successful training, the model's accuracy can be evaluated on validation data. Monitoring training and validation accuracy over epochs helps assess model performance.

Conclusion

This project showcases the potential of deep learning in preserving and recognizing ancient scripts. By applying modern AI techniques, the aim is to make historical texts accessible for research and education.
