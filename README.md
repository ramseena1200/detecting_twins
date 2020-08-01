# detecting_twins

Classification of twins using transfer learning

The aim of this project is to classify identical twins using Keras.Keras Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning.I have used VGG19,which is a variant of VGG model which in short consists of 19 layers (16 convolution layers, 3 Fully connected layer, 5 MaxPool layers and 1 SoftMax layer). There are other variants of VGG like VGG11, VGG16 etc. VGG19 has 19.6 billion FLOPs.

# Applications:
This project can be used in law and order if one of the identical twins is involved in crime,identifying and classifying identical twins which are very similar to each other etc.

# Requirements:
Keras API,Python 2.7 / jupyter notebook (python 2.7 compatible)

# Impelementation:
To implement my model, you may use "Predict_final_VGG19.py" script present in my repository.In this script you have to make a change in the following line of code-
data1=pd.read_csv("/home/risana/Downloads/Signal-Processing--master/5.Py_Scripts/Predict_Final.csv")

Actually the dataset of images of both twins I collected is stored in local drive of my computer.Then I used the VGG19 model to make mean vector for the image dataset of both the twins.

My training dataset consists of 127 images for each twin.The VGG19 model extract features from every images and return feature vector of each images while running the script.Then I computed mean feature vector for each categories.

While testing,I compared the feature vector of testing dataset with mean feature vector of each catagory using csine similarity and classified them as twin1 or twin2.

Obtained a fair accuracy while testing.
