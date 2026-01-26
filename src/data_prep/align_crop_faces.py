import cv2
import numpy as np

# -------------------------------
# Models
# -------------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("models/lbfmodel.yaml")

# -------------------------------
# Face alignment function
# -------------------------------
def align_face(
    img_path,
    output_size=(112, 112),
    margin=0.25
):
    """
    Aligns and crops a face for recognition / verification.

    Args:
        img_path (str): path to image
        output_size (tuple): (width, height)
        margin (float): extra crop margin around face

    Returns:
        aligned_face (np.ndarray) or None
    """

    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    if len(faces) == 0:
        return None

    # Take the largest detected face
    faces = sorted(
        faces, key=lambda b: b[2] * b[3], reverse=True
    )
    x, y, w, h = faces[0]

    success, landmarks = facemark.fit(img, np.array([faces[0]]))
    if not success:
        return None

    landmarks = landmarks[0][0]

    # -------------------------------
    # Eye centers
    # -------------------------------
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)

    # -------------------------------
    # Rotation
    # -------------------------------
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = ((left_eye + right_eye) / 2)
    eyes_center = (int(eyes_center[0]), int(eyes_center[1]))

    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1.0)

    rotated = cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]),
        flags=cv2.INTER_CUBIC
    )

    # -------------------------------
    # Rotate face box
    # -------------------------------
    bbox = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])

    ones = np.ones((4, 1))
    bbox = np.hstack([bbox, ones])
    rotated_bbox = M.dot(bbox.T).T

    x_min, y_min = rotated_bbox.min(axis=0).astype(int)
    x_max, y_max = rotated_bbox.max(axis=0).astype(int)

    # Add margin
    bw = x_max - x_min
    bh = y_max - y_min
    x_min -= int(bw * margin)
    y_min -= int(bh * margin)
    x_max += int(bw * margin)
    y_max += int(bh * margin)

    # Clip
    h_img, w_img = rotated.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w_img, x_max)
    y_max = min(h_img, y_max)

    face = rotated[y_min:y_max, x_min:x_max]

    if face.size == 0:
        return None

    # -------------------------------
    # Resize to model input
    # -------------------------------
    aligned_face = cv2.resize(face, output_size)

    return aligned_face

if __name__ == "__main__":
    
    result = align_face('././data/raw/family_adults/mum4.jpg')

    if result is not None:
        cv2.imshow("Aligned + Cropped(112,112)", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()