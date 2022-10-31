# https://www.tensorflow.org/lite/guide/python
# https://blog.paperspace.com/tensorflow-lite-raspberry-pi/
# https://github.com/raspberrypi/picamera2/blob/main/examples/capture_jp
# https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
# mobilenet quantization whitepaper https://arxiv.org/pdf/1704.04861.pdf

from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np
import time

from picamera2 import Picamera2, Preview


def load_labels(path): # Read the labels from the text file as a Python list.
    with open(path, 'r') as f:
        return [line.strip() for i, line in enumerate(f.readlines())]


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)

    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

    ordered = np.argpartition(-output, 1)
    return [(i, output[i]) for i in ordered[:top_k]][0]

data_folder = "mobile_net/mobilenet_v1_1.0_224_quant_and_labels/"

model_path = data_folder + "mobilenet_v1_1.0_224_quant.tflite"
label_path = data_folder + "labels_mobilenet_quant_v1_224.txt"

interpreter = Interpreter(model_path)
print("Model Loaded Successfully.")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Image Shape (", width, ",", height, ")")

picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})
picam2.configure(preview_config)

count = 1
cont = True
while cont:
    print("Press any key followed by enter to continue. Press Q or q to quit...")
    print("Place camera in front of image to classify")
    press = input()
    if press == 'q' or press == 'Q':
        print("Thank you, exiting program")
        cont = False
    else:
        # the main codeblock executes here
        picam2.start_preview(Preview.QTGL)
        picam2.start()

        # give the raspberry pi camera a few seconds to stabilize the image
        time.sleep(3)
        filename = ""
        if count < 10:
            filename = "photos/myphoto_0" + str(count) + ".jpg"
        else:
            filename = "photos/myphoto_" + str(count) + ".jpg"
        count += 1

        # take a picture and then stop the camera and stop the preview mode
        # this can be computationally expensive but the code is simpler
        picam2.capture_file(filename)

        # give the camera capturing thread time to fully complete before
        # switching off the preview mode and stopping the camera
        time.sleep(1)
        picam2.stop_preview()
        picam2.stop()

        # Load an image to be classified.
        # image = Image.open(data_folder + "cat02.jpg").convert('RGB').resize((width, height))
        image = Image.open(filename).convert('RGB').resize((width, height))

        # Classify the image.
        time1 = time.time()
        label_id, prob = classify_image(interpreter, image)
        time2 = time.time()
        classification_time = np.round(time2-time1, 3)
        print("Classificaiton Time =", classification_time, "seconds.")

        # Read class labels.
        labels = load_labels(label_path)

        # Return the classification label of the image.
        classification_label = labels[label_id]
        print("Image Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")
