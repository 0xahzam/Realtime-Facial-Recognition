import cv2
#load some pre-trained data on face frontals from open cv's haar cascade algorithm
#classifier is a fancy term for detector, here it will clasify something as a 'face'
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces, image is a two dimensional array, this is for importing the image
img = cv2.imread("Elon Musk.jpg")

#sonverting image to grayscale (B&W)cmd
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect the face, this will give us coordinates for rectangle to be drawn
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#plotting the rectangle
for x,y,w,h in face_coordinates:
    cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0),2) #opencv is BGR not RGB, 255 is for green, 2 is thin line


cv2.imshow("Face Detector",img)

cv2.waitKey()


