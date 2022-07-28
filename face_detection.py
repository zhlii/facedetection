import os
import numpy as np
import cv2 as cv


def resize_image_keep_ratio(image, target_size):
    old_size = image.shape[0:2]
    radio = min(float(target_size[i])/(old_size[i])
                for i in range(len(old_size)))

    new_size = tuple([int(i*radio) for i in old_size])

    new_image = cv.resize(image, (new_size[1], new_size[0]))

    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]

    top, bottom = pad_h//2, pad_h-(pad_h//2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    return cv.copyMakeBorder(new_image, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))


def main():
    directory = os.path.dirname(__file__)
    capture = cv.VideoCapture(1)

    if not capture.isOpened():
        exit()

    weights = os.path.join(
        directory, "yunet_yunet_final_640_640_simplify.onnx")

    face_detector = cv.FaceDetectorYN_create(weights, "", (0, 0))

    while True:
        result, image = capture.read()
        if result is False:
            cv.waitKey(0)
            break

        channels = 1 if len(image.shape) == 2 else image.shape[2]

        if channels == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

        image = resize_image_keep_ratio(image, (640, 640))

        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        for face in faces:
            box = list(map(int, face[:4]))
            color = (0, 0, 255)
            thickness = 1
            cv.rectangle(image, box, color, thickness, cv.LINE_AA)

            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            for landmark in landmarks:
                radius = 5
                thickness = -1
                cv.circle(image, landmark, radius,
                          color, thickness, cv.LINE_AA)

            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            cv.putText(image, confidence, position, font,
                       scale, color, thickness, cv.LINE_AA)

        cv.imshow("face detection", image)
        key = cv.waitKey(10)
        if key == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
