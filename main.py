import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
# https://developers.google.com/mediapipe/solutions/vision/object_detector/python

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

model_path = os.path.join(os.getcwd(), "efficientdet_lite0.tflite")

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image



# model_path = r"C:\Users\Sai\Work\OpenCVTest\efficientdet_lite0.tflite"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # STEP 3: Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # STEP 4: Convert the BGR frame to RGB
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # STEP 5: Create a MediaPipe Image object from the frame
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # STEP 6: Detect objects in the frame
    # detection_result = detector.detect_async(image, 0)

    # STEP 7: Process the detection result. In this case, visualize it.
    image_copy = np.copy(frame)

    # annotated_image = visualize(image_copy, detection_result)
    # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Objecct Detection',image_copy)
    # cv2.imshow('Objecct Detection',annotated_image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
