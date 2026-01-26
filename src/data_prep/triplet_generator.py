import os
import random
import cv2
import numpy as np

class TripletGenerator:
    def __init__(self, processed_dir, batch_size=32):
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        
        # Paths to your subfolders
        self.nephew_dir = os.path.join(processed_dir, 'nephew')
        self.family_dir = os.path.join(processed_dir, 'family_adults')
        self.stranger_dir = os.path.join(processed_dir, 'stranger_adults')
        
        # Lists of image paths
        self.nephew_imgs = self._get_img_list(self.nephew_dir)
        self.family_imgs = self._get_img_list(self.family_dir)
        self.stranger_imgs = self._get_img_list(self.stranger_dir)

    def _get_img_list(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def _load_and_preprocess(self, path):
        # MobileFaceNet standard: Load, normalize to [-1, 1]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 127.5) / 128.0
        return img

    def get_batch(self):
        """Generates a balanced batch of (Anchor, Positive, Negative)"""
        while True:
            anchors, positives, negatives = [], [], []
            
            for _ in range(self.batch_size):
                # 1. Anchor & Positive (Always Nephew)
                # Ensure they are different images
                a_path, p_path = random.sample(self.nephew_imgs, 2)
                
                # 2. Negative (50% Family, 50% Strangers)
                # This makes it "Semi-Hard" for the model
                if random.random() > 0.5:
                    n_path = random.choice(self.family_imgs)
                else:
                    n_path = random.choice(self.stranger_imgs)
                
                anchors.append(self._load_and_preprocess(a_path))
                positives.append(self._load_and_preprocess(p_path))
                negatives.append(self._load_and_preprocess(n_path))
                
            yield [np.array(anchors), np.array(positives), np.array(negatives)]

# Standalone Test logic
if __name__ == "__main__":
    gen = TripletGenerator('data/processed')
    batch = next(gen.get_batch())
    print(f"Generated batch with {len(batch[0])} triplets.")
    print(f"Anchor shape: {batch[0].shape}") # Should be (batch_size, 112, 112, 3)