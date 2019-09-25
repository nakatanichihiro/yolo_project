import cv2

# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(-1)

while(True):
    ret, frame = capture.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()