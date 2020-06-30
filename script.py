import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

img = cv2.imread("photo.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.equalizeHist(gray_img)
faces = face_cascade.detectMultiScale(gray_img,
scaleFactor = 1.1,
minNeighbors = 5
)

# for x, y, h, w in faces:
#     img = cv2.rectangle(img, (x,y), (x+h, y+w), (0, 255, 0), 3)

for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        img = cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360, (0, 255, 0), 4)
        faceROI = gray_img[y:y+h,x:x+w]
        
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI, scaleFactor=1.035)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            img = cv2.circle(img, eye_center, radius, (255, 0, 0 ), 4)


img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
cv2.imshow("Hi", img)
cv2.waitKey(0)
cv2.destroyAllWindows()