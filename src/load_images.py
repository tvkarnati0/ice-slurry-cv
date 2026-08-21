import cv2
from pathlib import Path

def load_images(folder_path: str="images", pattern: str="*.png"):
    folder=Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")
 
    image_paths = sorted(folder.glob(pattern))
 
    if not image_paths:
        print(f"No files matching '{pattern}' found in {folder.resolve()}")
        return []
    loaded = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Warning: failed to load {path.name}")
            continue
        loaded.append((path.name, img))
 
    return loaded
 
 
if __name__ == "__main__":
    images = load_images()
 
    print(f"\nLoaded {len(images)} image(s):")
    for filename, img in images:
        print(f"  {filename} -> shape {img.shape}")

def convert_to_grayscale(images: list) -> list:
    grayscale = []
    for filename, img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale.append((filename, gray_img))

    return grayscale

if __name__ == "__main__":
    images = load_images()
 
    print(f"\nLoaded {len(images)} image(s):")
    for filename, img in images:
        print(f"  {filename} -> shape {img.shape}")

    gray_images = convert_to_grayscale(images)

    print(f"\nConverted {len(gray_images)} image(s) to grayscale:")
    for filename, gray_img in gray_images:
        print(f" {filename} -> shape {gray_img.shape}")
        cv2.imshow('Grayscale',gray_img)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

rim_petri_dish = (370, 947, 380, 965)

def get_brightness(gray_img, box):
    x1, y1, x2, y2 = box
    patch = gray_img[y1:y2, x1:x2]
    return patch.mean()

for filename, gray_img in gray_images:
    brightness = get_brightness(gray_img, rim_petri_dish)
    print(f"{filename}: rim brightness = {brightness:.2f}")

ice_roi = (435, 1001, 906, 1382)

def get_brightness(gray_img, box2):
    x1, y1, x2, y2 = box2
    patch = gray_img[y1:y2, x1:x2]
    return patch.mean()

for filename, gray_img in gray_images:
    ice_brightness = get_brightness(gray_img, ice_roi)
    print(f"{filename}: ice brightness = {ice_brightness:.2f}")

time_interval = 15
times = [i*time_interval for i in range(len(gray_images))]

results = []
for i, (filename, gray_img) in enumerate(gray_images):
    ice_brightness = get_brightness(gray_img, ice_roi)
    time_elapsed = times[i]
    results.append((time_elapsed, ice_brightness))
    print(f"t={time_elapsed}s | {filename}: ice brightness = {ice_brightness:.2f}")

import matplotlib.pyplot as plt
times_list, brightness_list = zip(*results)
plt.plot(times_list, brightness_list, marker="o")
plt.xlabel("Time (seconds)")
plt.ylabel("Mean Grayscale Intensity (Ice ROI)")
plt.title("Ice Slurry Melting Progression")
plt.show()