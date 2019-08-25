from LeukemiaModel import leukemia_cnn
from skimage.transform import resize
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16

classes = [
    'Limfoblas',
    'Limfosit',
    'Metamielosit',
    'Monosit',
    'Segment',
    'Stab'
]

img = plt.imread('Limfoblas.jpg')
test_img = np.array(img)

a = resize(test_img, preserve_range=True, output_shape=(200,200)).astype(int)
test_image = np.array(a)

# preprocessing the images
test_image = preprocess_input(test_image)

# Create base_model
base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (200, 200, 3))

# extracting features from the images using pretrained model
test_image = base_model.predict(test_image.reshape(1,200,200,3))

# zero centered images
test_image = test_image/test_image.max()

# make prediction lists
predictions = model.predict(test_image.reshape(1,6*6*512))

# print the result and show the image
top_3 = np.argsort(predictions[0])[:-4:-1]
plt.imshow(img)
for i in range(3):
    print("{}".format(classes[top_3[i]])+" ({:.3})".format(predictions[0][top_3[i]]))