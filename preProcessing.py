import cv2
import pytesseract
from PIL import Image
import operator
import os

def main():
        # Get File Name from Command Line
        path = input("Enter the file path : ").strip()
        # load the image
        image = cv2.imread(path)
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (2,2))
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        filename = "{}.png".format("temp")

        cv2.imwrite(filename, gray)

def judge_value(list_string):
    dic_count = {}
    dic_count["100"] = list_string.count("100")
    dic_count["50"] = list_string.count("50")
    dic_count["20"] = list_string.count("20")
    dic_count["10"] = list_string.count("10")
    dic_count["5"] = list_string.count("5")
    dic_count["1"] = list_string.count("1")
    print("dic: ", dic_count)
    if dic_count["100"] == 0 and dic_count["50"] == 0 and dic_count["20"] == 0 and dic_count["10"] == 0 and dic_count["5"] == 0 and dic_count["1"] == 0:
        print("undefined")
        return
    face_value = max(dic_count.items(), key=operator.itemgetter(1))[0]
    print(">>>>>>>>>>", face_value, "banknote")
    
main()