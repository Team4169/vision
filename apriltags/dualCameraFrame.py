import cv2

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

print(cap0.read()[1].shape)
print(cap1.read()[1].shape)
print(cap2.read()[1].shape)

quit()

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    cv2.imshow("Camera 0", frame0)
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()
