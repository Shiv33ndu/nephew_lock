import os
import random
import shutil

# Example logic for your triplet_generator.py prep
def collect_strangers(lfw_path, target_dir, count=60):
    all_people = os.listdir(lfw_path)
    selected_pics = []
    
    while len(selected_pics) < count:
        person = random.choice(all_people)
        person_dir = os.path.join(lfw_path, person)
        pic = random.choice(os.listdir(person_dir))
        selected_pics.append(os.path.join(person_dir, pic))
        
    # Copy to your project's negative folder
    for i, p in enumerate(selected_pics):
        shutil.copy(p, os.path.join(target_dir, f"stranger_{i}.jpg"))



if __name__ == "__main__":

    path_to_pick_from = ".\\data\\lfw"
    target_path = ".\\data\\raw\\stranger_adults"
    
    collect_strangers(path_to_pick_from, target_path)