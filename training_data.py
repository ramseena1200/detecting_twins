from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump
from os import listdir
from matplotlib import image
# load model
model = VGG16()

# remove the output layer
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
# load all images in a directory

loaded_images = list()
# load an image from file
for filename in listdir('C:/Users/DELL/Desktop/ramsi'):
    #load image
    image = load_img('C:/Users/DELL/Desktop/ramsi/' + filename , target_size=(224,224))
   
    #convert the image pixels to a numpy array
    image = img_to_array(image)
    #reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    #get extracted features
    features = model.predict(image)
    print(features.shape)
    #save to file
    dump(features, open('C:/Users/DELL/Desktop/ramsi_pkl/'+ filename + '.pkl', 'wb'))
