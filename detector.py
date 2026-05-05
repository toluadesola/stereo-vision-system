import cv2
import numpy as np
import onnxruntime as ort
from config import *

class YOLODetector:

    def __init__(self, model_path, class_file):

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        with open(class_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect(self, frame):

        h, w = frame.shape[:2]

        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = self.session.run(None, {self.input_name: img})
        preds   = np.squeeze(outputs[0])

        boxes       = []
        confidences = []
        class_ids   = []

        for pred in preds:

            x, y, bw, bh, obj = pred[:5]
            scores   = pred[5:]
            class_id = int(np.argmax(scores))
            conf     = float(obj * scores[class_id])

            if conf < CONF_THRESHOLD:
                continue

            # Convert from centre format to top-left corner format (pixels)
            x1 = int((x  - bw / 2) * w / INPUT_SIZE)
            y1 = int((y  - bh / 2) * h / INPUT_SIZE)
            bw = int(bw * w / INPUT_SIZE)
            bh = int(bh * h / INPUT_SIZE)

            boxes.append([x1, y1, bw, bh])
            confidences.append(conf)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD
        )

        results = []
        for i in indices:
            idx = int(i)
            results.append({
                "box":        boxes[idx],
                "label":      self.classes[class_ids[idx]],
                "confidence": confidences[idx]
            })

        return results