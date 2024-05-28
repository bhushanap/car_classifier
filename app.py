import gradio as gr
from fastai.vision.all import *
# import os
# Load a pre-trained image classification model
import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

root = os.path.dirname(__file__)



def get_label(fname):
    id = int(fname.name[-9:-4])
#     print(id)
    cls = int(labels[id-1])-1
#     print(cls)
    return name(cls)

learn = load_learner("./models/model.pkl")


# Function to make predictions from an image
def classify_image(image):
    # Make a prediction
    # Decode the prediction and get the class name
    name = learn.predict(image)
    return name[0]

# Sample images for user to choose from
sample_images = ["./sample_images/AcuraTLType-S2008.jpg", "./sample_images/AudiR8Coupe2012.jpg","./sample_images/DodgeMagnumWagon2008.jpg"]

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(label="Select an image", type="filepath"),
    outputs="text",
    live=False,
    title="Car image classifier",
    description="Upload a car image or select one of the examples below",
    examples=sample_images
)


iface.launch()