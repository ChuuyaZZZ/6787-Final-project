import os
import cv2 as cv


def SIFT():
    for i in range(7):
        name = "data/test" + str(i) + ".jpg"
        print(name)
        original_Image = cv.imread(name)
        descriptor = cv.xfeatures2d.SIFT_create(4000)
        kp_original, des_original = descriptor.detectAndCompute(original_Image, None)
        flann = cv.BFMatcher()

        for eachphoto in os.listdir("data/england"):
            image = cv.imread("data/england/" + eachphoto)
            kp_curr, des_curr = descriptor.detectAndCompute(image, None)

            matches_middle = flann.knnMatch(des_original, des_curr, k=2)
            matches_count = 0
            # Apply ratio test
            for m, n in matches_middle:
                if m.distance < 0.5 * n.distance:
                    matches_count += 1

            print(matches_count)
            if matches_count > 50:
                print(eachphoto)