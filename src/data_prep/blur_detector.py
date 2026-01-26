import os
import cv2

def check_blur(img_path, threshold = 100.0):
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return variance < threshold, variance


if __name__ == "__main__":

    # check blurry score
    score = check_blur('.\\.\\data\\raw\\nephew\\nephew (3).jpg')
    # score = check_blur('.\\.\\tests\\blur_check.jpg')

    print(f"Blurry: {score[0]} (score: {score[1]})")