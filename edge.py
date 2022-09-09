import cv2
import numpy as np

img = cv2.imread("shape.jpeg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur= cv2.blur(gray, (6,6))

ret,thresh = cv2.threshold(blur,160,255,cv2.THRESH_BINARY)
cv2.imwrite("binary.jpeg",thresh)

# inverted_binary = ~ binary

# cv2.imshow("binary",thresh)
# cv2.waitKey(0)

median = np.median(blur)
print(median)

low = int(max(0, (1 - 0.33)*median))
high = int(min(255, (1 + 0.33)*median))

edges = cv2.Canny(image = thresh, threshold1 = low, threshold2 = high)
# cv2.imshow("edges",edges)
# cv2.waitKey(0)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

for c in contours:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.drawContours(thresh, [c], -1, (0, 255, 0), 2)
    cv2.circle(thresh, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(thresh, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("Image", thresh)
    cv2.waitKey(0)

for i in contours:
    img = cv2.imread("binary.jpeg")
    size = cv2.contourArea(i)
    rect = cv2.minAreaRect(i)
    if size > 10000:
        gray = np.float32(gray)
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.fillPoly(mask, [i], (255, 255, 255))
        dst = cv2.cornerHarris(mask, 18, 3, 0.09)
        ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
        if rect[2] == 0 and len(corners) == 5:
            x, y, w, h = cv2.boundingRect(i)
            if w == h or w == h + 3:  # Just for the sake of example
                # print('Square corners: ')
                for i in range(1, len(corners)):
                    print(corners[i])
                cv2.put

            else:
                print('Rectangle corners: ')
                for i in range(1, len(corners)):
                    print(corners[i])
        if len(corners) == 5 and rect[2] != 0:
            print('Rombus corners: ')
            for i in range(1, len(corners)):
                print(corners[i])
        if len(corners) == 4:
            print('Triangle corners: ')
            for i in range(1, len(corners)):
                print(corners[i])
        if len(corners) == 6:
            print('Pentagon corners: ')
            for i in range(1, len(corners)):
                print(corners[i])
        img[dst > 0.1 * dst.max()] = [0, 0, 255]
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows


