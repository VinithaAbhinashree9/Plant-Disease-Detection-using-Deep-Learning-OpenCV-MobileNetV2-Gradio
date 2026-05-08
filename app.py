import gradio as gr
import tensorflow as tf
import numpy as np
import cv2



model = tf.keras.models.load_model("plant_disease_model.h5")



class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]



IMG_SIZE = 128



def predict_disease(image):

    # Resize image
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img = img / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction)

    return f"""
🌱 Prediction: {predicted_class}

📊 Confidence: {confidence:.2f}
"""



interface = gr.Interface(

    fn=predict_disease,

    inputs=gr.Image(type="numpy"),

    outputs="text",

    title="🌱 Plant Disease Detector AI",

    description="Upload a plant leaf image to detect disease"

)



interface.launch()