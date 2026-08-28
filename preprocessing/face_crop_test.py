import cv2
import os

# --------------------------------------------------
# INPUT IMAGE
# --------------------------------------------------

INPUT_FOLDER = r"D:\STREESHIELD_Dataset\train\real"

# Get the first image from the folder
image_files = [
    file for file in os.listdir(INPUT_FOLDER)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    print("No image found.")
    exit()

input_filename = image_files[0]
input_path = os.path.join(INPUT_FOLDER, input_filename)

print("Input image:", input_path)


# --------------------------------------------------
# OUTPUT LOCATION
# --------------------------------------------------

OUTPUT_FOLDER = r"D:\STREESHIELD_Dataset\processed\train\real"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

output_path = os.path.join(
    OUTPUT_FOLDER,
    input_filename
)


# --------------------------------------------------
# READ IMAGE
# --------------------------------------------------

image = cv2.imread(input_path)

if image is None:
    print("Could not read image.")
    exit()


# --------------------------------------------------
# LOAD HAAR CASCADE
# --------------------------------------------------

cascade_path = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("Could not load Haar Cascade.")
    exit()


# --------------------------------------------------
# CONVERT TO GRAYSCALE
# --------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------
# DETECT FACE
# --------------------------------------------------

faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Faces detected:", len(faces))


# --------------------------------------------------
# CHECK WHETHER FACE WAS FOUND
# --------------------------------------------------

if len(faces) == 0:
    print("No face detected.")
    exit()


# --------------------------------------------------
# SELECT THE LARGEST FACE
# --------------------------------------------------

largest_face = max(
    faces,
    key=lambda face: face[2] * face[3]
)

x, y, w, h = largest_face

print("Face coordinates:")
print("x =", x)
print("y =", y)
print("width =", w)
print("height =", h)


# --------------------------------------------------
# CROP FACE
# --------------------------------------------------

face = image[y:y + h, x:x + w]


# --------------------------------------------------
# RESIZE TO 128 x 128
# --------------------------------------------------

face_resized = cv2.resize(
    face,
    (128, 128)
)


# --------------------------------------------------
# SAVE PROCESSED FACE
# --------------------------------------------------

success = cv2.imwrite(
    output_path,
    face_resized
)

if success:
    print("Processed face saved to:")
    print(output_path)
else:
    print("Failed to save processed image.")