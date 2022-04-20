import cv2
#live detection via webcamimport cv2

#load some pre-trained data on face frontals from open cv's haar cascade algorithm
#classifier is a fancy term for detector, here it will clasify something as a 'face'
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#capture video, 0/1/2 indicates from webcam
webcam = cv2.VideoCapture(0)

#iterate forever over frames
while True:

    #read the current frame
    successful_frame_read, frame = webcam.read(0)

    # convert frame to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect the face, this will give us coordinates for rectangle to be drawn
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for x,y,w,h in face_coordinates:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2) 

    cv2.imshow("Live Face Detector",frame)
    
    #waitkey prevents the window from closing instantaneously, press Q to quit

    key = cv2.waitKey(1)

    if key==81 or key==113:
        break
    
