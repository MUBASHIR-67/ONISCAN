import cv2

image_path = "dataset/raw/Onion-Bad.jpg"

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load image")
else:
    height, width, channels = image.shape

    print("Computer Vision Test Successful!")
    print("Width:", width)
    print("Height:", height)
    print("Channels:", channels)
    