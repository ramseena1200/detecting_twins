from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump
from os import listdir
from matplotlib import image
import scipy
from scipy import spatial
import pickle
# load model
model = VGG16()
# remove the output layer
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
# load all images in a directory
vector1 = pickle.load((open("C:/Users/DELL/Desktop/risuvgg16_mean.pkl",'rb')))
vector2 = pickle.load((open("C:/Users/DELL/Desktop/ramsivgg16_mean.pkl",'rb')))
loaded_images = list()
# load an image from file
for filename in listdir('C:/Users/DELL/Desktop/vtoframe'):
    # load image
    image = load_img('C:/Users/DELL/Desktop/vtoframe/' + filename , target_size=(224,224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get extracted features
    features = model.predict(image)
    print(features.shape)
    cosine_similarity_1 = 1 - spatial.distance.cosine(features, vector1)
    print(cosine_similarity_1)
    cosine_similarity_2 = 1 - spatial.distance.cosine(features, vector2)
    print(cosine_similarity_2)
    if cosine_similarity_1>cosine_similarity_2:
        dump(features, open('C:/Users/DELL/Desktop/testing/risu/'+ filename + '.pkl', 'wb'))
    else :
        dump(features, open('C:/Users/DELL/Desktop/testing/ramsi/'+ filename + '.pkl', 'wb'))
