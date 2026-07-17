import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['en'])

def recognize_text(png_bytes):

    image = cv2.imdecode(
        np.frombuffer(png_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )
    _, gray = cv2.threshold(
       gray,
       127,
       255,
       cv2.THRESH_BINARY
    )

    result = reader.readtext(gray)

    if not result:
        return ""

    text = " ".join(
        [item[1] for item in result]
    )

    return text