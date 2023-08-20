import base64

import cv2

from face_recognition import register
from flask import Flask, request
app = Flask(__name__)


@app.route('/imageToId', methods=['POST'])
def imageToId():
    print("image name")
    image = request.data
    image_data = base64.b64decode(image)

    # Get the path to the current directory
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    # # Construct the path to the data directory
    # data_directory = os.path.join(current_directory, '../data')
    # print("Image_name:", image_name)
    # input_image_path = os.path.join(data_directory, image_name + ".png")
    #result = register([image_data])
    # cv2.imshow("njk",image_data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("image data", image_data)
    return {
        "words": 1234
    }