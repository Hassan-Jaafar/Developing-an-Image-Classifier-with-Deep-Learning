import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import argparse
batch_size = 32
image_size = 224

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()
    
def predict (image_path ,model,top_k=5 ):
    loaded_image = Image.open(image_path)
    processed_image = process_image(np.asarray(loaded_image))
    expanded_image =  np.expand_dims(processed_image,axis =0)
    pred_image = model.predict(expanded_image)
    values,indices = tf.nn.top_k(pred_image, k=top_k)
    probs = values.numpy()[0]
    classes = indices.numpy()[0] + 1 
    flower_names = []
    for i in classes:
        for k in class_names:
            if str(i) == k :
                flower_names.append(class_names[k])
     
    
    return probs ,classes , processed_image,flower_names
     
    
                                  
                                  
if __name__ == '__main__':
    print('predict.py, running')    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names') 
    args = parser.parse_args()
    print(args)
    print('arg1:', args.arg1)
    print('arg2:', args.arg2)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    image_path = args.arg1
    model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = args.top_k
    if top_k is None: 
        top_k = 5
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    probs, classes , image , flower_names = predict(image_path, model, top_k)    
    
    for i in range(top_k):
        print( "flower name is {} and its probabilty  {} \n".format(flower_names[i] , probs[i] ))
        
    
     
    
                                      