# IMPORTS
import cv2
import PySimpleGUI as sg
from tensorflow import keras as K
import numpy as np

# PLANT & DISEASE NAMES
class_names_master = ["APPLE_BLACK_ROT", "APPLE_CEDAR_RUST", "APPLE_HEALTHY", "APPLE_SCAB", "BACKGROUND",
                      "BELLPEPPER_BACTERIAL_SPOT",
                      "BELLPEPPER_HEALTHY", "CHERRY_HEALTHY", "CHERRY_POWDERY_MILDEW", "CITRUS_BLACK_SPOT",
                      "CITRUS_CANKER", "CITRUS_GREENING", "CITRUS_HEALTHY", "CORN_COMMON_RUST", "CORN_GREY_LEAF_SPOT",
                      "CORN_HEALTHY", "CORN_NORTHERN_LEAF_BLIGHT", "GRAPE_BLACK_MEASLES", "GRAPE_BLACK_ROT",
                      "GRAPE_HEALTHY", "GRAPE_ISARIOPSIS_LEAF_SPOT", "PEACH_BACTERIAL_SPOT", "PEACH_HEALTHY",
                      "POTATO_EARLY_BLIGHT", "POTATO_HEALTHY", "POTATO_LATE_BLIGHT", "STRAWBERRY_HEALTHY",
                      "STRAWBERRY_LEAF_SCORCH", "TOMATO_BACTERIAL_SPOT", "TOMATO_EARLY_BLIGHT", "TOMATO_HEALTHY",
                      "TOMATO_LATE_BLIGHT", "TOMATO_LEAF_MOLD", "TOMATO_MOSAIC_VIRUS", "TOMATO_SEPTORIA_LEAF_SPOT",
                      "TOMATO_SPIDER_MITES", "TOMATO_TARGET_SPOTS", "TOMATO_YELLOW_LEAF_CURL_VIRUS"]

# PLANT DISEASE DETECTION MODELS
master_model = K.models.load_model("Models/MASTER/MASTER_Model.h5")

# CONSTANTS
disease = 5
confidence = 1
webcamNum = 0

# APPLICATION GUI
headings_font = ('Courier New', 24)
gui_layout = [[sg.Titlebar('PLANT DISEASE DETECTION 1.0')],
              [sg.Column([[sg.Image(filename='', key='image')]], justification="center")],
              [sg.Column([[sg.Text('PREDICTION PROBABILITY', size=(32, 1), font=('Helvetica', 20), justification='center')]],
                         justification="center")],
              [sg.Column([[sg.Text('', size=(32, 1), font=('Helvetica', 20), justification='center', key='text')]],
                         justification="center")]]
sg.theme('Dark')


def get_result(frame, trained_model):
    imgResize = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    test_image_array = np.array([imgResize])
    result = trained_model.predict(test_image_array)
    pred = np.argmax(result)
    prob = np.amax(result)
    return pred, prob


def results_display_disease(result, img, class_name):
    text = "None"
    if not None:
        text = "Prediction: " + class_name[result]
    img = cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2, cv2.LINE_AA)
    return img


def main():
    global disease, confidence
    i = 4
    window, cap = sg.Window('PLANT DISEASE DETECTION', gui_layout, keep_on_top=True), cv2.VideoCapture(webcamNum)
    while True:
        event, values = window.read(timeout=20)
        if event == sg.WIN_CLOSED:
            return

        # VIDEO CAPTURE
        ret, frame = cap.read()

        # RUN CHOSEN MODEL
        i += 1
        if i % 5 == 0:
            disease, confidence = get_result(frame, master_model)
        img = results_display_disease(disease, np.array(frame), class_names_master)
        img = cv2.resize(img, (600, 600))

        # VIDEO DISPLAY
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        window['image'].update(data=imgbytes)

        # GUI Update Prediction
        window['text'].update(class_names_master[disease] + " " + str(round(confidence * 100, 2)) + "%")


if __name__ == '__main__':
    main()
