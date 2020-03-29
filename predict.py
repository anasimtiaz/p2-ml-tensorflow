import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse

# Set up argparse
parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to image")
parser.add_argument("saved_model", help="Path to saved model")
parser.add_argument("--top_k", help="Return the top K most likely classes", type = int)
parser.add_argument("--category_names", help="Path to a JSON file mapping labels to flower names")
args = parser.parse_args()

# read json file if argument provided
if args.category_names is not None:
    with open(args.category_names, 'r') as f:
        category_names = json.load(f)

if args.top_k is None:
    i_top_k = 1
else:
    i_top_k = args.top_k

# Load saved model
saved_model = tf.keras.models.load_model(args.saved_model, custom_objects={'KerasLayer':hub.KerasLayer})
saved_model.summary()


#

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis = 0)
    prediction = model.predict(processed_test_image)
    probs, classes = tf.math.top_k(prediction, k=top_k)
    return probs, classes


probs, classes = predict(args.image, saved_model, i_top_k)
print(probs.numpy())
print(classes.numpy())

if args.category_names is not None:
    names = []
    for i in range(i_top_k):
        names.append(category_names[(classes.numpy()[0,i]+1).astype(str)])

    print(category_names[(classes.numpy()[0,0]+1).astype(str)])
    print(probs.numpy()[0,0])
