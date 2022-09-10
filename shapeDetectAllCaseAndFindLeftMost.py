import cv2
import numpy as np
import imutils

img = cv2.imread("rotate.jpeg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur= cv2.blur(gray, (6,6))
print(img.shape)

ret,thresh = cv2.threshold(blur,158,255,cv2.THRESH_BINARY)
# cropped = thresh[21:1060,:]
# cropped = cv2.resize(cropped,(1440,1080))
cv2.imshow("cropped",thresh)
print(thresh.shape)
cv2.waitKey(0)
cv2.imwrite("binary.jpeg",thresh)

# inverted_binary = ~ binary

# cv2.imshow("binary",thresh)
# cv2.waitKey(0)

median = np.median(blur)
print(median)

low = int(max(0, (1 - 0.33)*median))
high = int(min(255, (1 + 0.33)*median))

edges = cv2.Canny(thresh, low, high)
edges = cv2.dilate(edges, None, iterations=1)
edges = cv2.erode(edges, None, iterations=1)

cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


cv2.imshow("blur",edges)
cv2.waitKey(0)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

list = []
list2 = []

for c in contours:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    list.append(cX)
    list2.append(cY)
minX = min(list)
print(minX)

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

    size = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)

    thresh = np.float32(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    cv2.fillPoly(mask, [c], (255, 255, 255))
    dst = cv2.cornerHarris(mask, 20, 3, 0.09)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(thresh, np.float32(centroids), (5, 5), (-1, -1), criteria)

    if (len(corners) == 2):
        if (cX == minX):
            cv2.putText(img, "Top left : Circle", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            continue
        cv2.putText(img, "Circle", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if (len(corners) == 4):
        if (cX == minX):
            cv2.putText(img, "Top left : Triangle", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            continue
        cv2.putText(img, "Triangle", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if (len(corners) == 5):
        print("1 : ",np.array(corners[1]))
        print("2 : ",np.array(corners[2]))
        print("3 : ",np.array(corners[3]))

        nar1 = abs(np.array(corners[1]) - np.array(corners[2]))
        nar2 = abs(np.array(corners[1]) - np.array(corners[3]))
        print("nar1 : ",nar1)
        print("nar2 : ", nar2)

        edge1 = (nar1[0] ** 2 + nar1[1] ** 2) ** 0.5
        edge2 = (nar2[0] ** 2 + nar2[1] ** 2) ** 0.5
        print("edge1 : ",edge1)
        print("edge2 : ", edge2)

        aspectRatio = float(edge1/edge2)


        #x, y, w, h = cv2.boundingRect(c)
        #print(x,y,w,h)
        #aspectRatio = float(w)/h
        if aspectRatio >= 0.90 and aspectRatio < 1.10:
            print("aspect ratio : ", aspectRatio)
            for i in range(1,len(corners)):
                print(corners[i])
            if (cX == minX):
                cv2.putText(img, "Top left : Square", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                continue
            cv2.putText(img, "Square", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            if (cX == minX):
                cv2.putText(img, "Top left : Rectangle", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                continue
            cv2.putText(img, "Rectangle", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if (len(corners) == 6):
        if (cX == minX):
            cv2.putText(img, "Top left : Pentagon", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            continue
        cv2.putText(img, "Pentagon", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if (len(corners) == 7):
        if (cX == minX):
            cv2.putText(img, "Top left : Hexagon", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            continue
        cv2.putText(img, "Hexagon", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imshow("Shape",img)
cv2.waitKey(0)






