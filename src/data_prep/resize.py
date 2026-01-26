import cv2


def resize_only(img_path, output_size=(112, 112)):
    img = cv2.imread(img_path)
    return cv2.resize(img, output_size)


if __name__ == "__main__":
    result = resize_only('././data/raw/nephew/nephew (42).jpg')

    if result is not None:
        cv2.imshow("Cropped(112,112)", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()