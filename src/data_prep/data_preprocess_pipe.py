import os
import cv2
from src.data_prep.align_crop_faces import align_face
from src.data_prep.blur_detector import check_blur
from src.data_prep.resize import resize_only


def data_preprocess_pipe(dir_path, only_resize = False):
    """
    preprocess the images in given folder path
    - detect blurred images and skip them
    - align the face and crop it in dim compatible for MobileFaceNet input  
    - store the files into processed/dir_path 
    
    :param dir_path: dir_path 
    """

    dir_path = os.path.abspath(dir_path)

    print(dir_path)

    # define processed root
    processed_root = dir_path.replace("raw", "processed")

    print(processed_root)

    if processed_root == dir_path:
        raise ValueError("Expected 'raw' in directory path")
    
    os.makedirs(processed_root, exist_ok=True)

    total, saved, skipped_blur, skipped_align = 0, 0, 0, 0

    skipped_blur_files = []
    skipped_face_align_files = []

    # walking through the directory tree
    for root, _, files in os.walk(dir_path):

        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            total += 1

            img_path = os.path.join(root, file)

            # read image 
            img = cv2.imread(img_path)

            if img is None:
                continue

            # detect blur 
            is_blurry, _ = check_blur(img_path, 12.0)  # intentional, to let pass the imperfect low quality phone pictures

            if is_blurry:
                skipped_blur += 1
                skipped_blur_files.append(os.path.basename(img_path))
                continue

            # align and crop into 112x112 dim logic

            # if resize only chosen then we dont align the image, just resize it in 112x112  
            if only_resize:
                final_face = resize_only(img_path)
            
            # else we align and then resize the image to 112x112
            else:
                final_face = align_face(img_path)

                if final_face is None:
                    skipped_face_align_files.append(os.path.basename(img_path)) # to track the filenames
                    skipped_align += 1
                    continue

            
            # build output path
            rel_path = os.path.relpath(root, dir_path)
            save_dir = os.path.join(processed_root, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, file)

            cv2.imwrite(save_path, final_face)
            saved += 1
     
    print("Preprocessing complete")
    print(f"Total images     : {total}")
    print(f"Saved            : {saved}")
    print(f"Skipped (blur)   : {skipped_blur}")
    print(f"---skipped Files--\n{' ,'.join(skipped_blur_files)}")
    
    print(f"\n\nSkipped (no face): {skipped_align}")
    print(f"--skipped files--\n{' ,'.join(skipped_face_align_files)}")


if __name__ == "__main__":

    data_preprocess_pipe(".\\.\\data\\raw", only_resize=True)