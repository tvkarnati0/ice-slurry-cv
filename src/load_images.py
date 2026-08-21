"""
Ice slurry melting analysis pipeline.

Two parallel methods are run:
  1. Box method       -- plain rectangular ROI mean intensity
  2. Fixed-mask method -- user hand-sketches the slurry outline on
                          frame 1 (mouse drag in a popup window),
                          that shape is locked in and reused on every
                          frame to track brightness + ice-fraction.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------- Load ----------
def load_images(folder_path: str = "images", pattern: str = "*.png"):
    folder = Path(folder_path)

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


# ---------- Grayscale ----------
def convert_to_grayscale(images: list) -> list:
    grayscale = []
    for filename, img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale.append((filename, gray_img))
    return grayscale


# ---------- Plain rectangular brightness (box method + rim check) ----------
def get_brightness(gray_img, box):
    x1, y1, x2, y2 = box
    patch = gray_img[y1:y2, x1:x2]
    return patch.mean()


# ---------- Interactive freehand sketch -> filled mask ----------
def sketch_mask(first_gray_img, box, save_path="output/fixed_mask_sketch.png", target_size=1200):
    """
    Opens a window showing frame 1's ROI, upscaled so its largest
    dimension is roughly `target_size` pixels, for easier sketching.
    Click and drag to sketch an outline around the slurry. Press any
    key when done. Returns the filled-in binary mask, sized to match
    the ORIGINAL (non-upscaled) ROI dimensions.
    """
    x1, y1, x2, y2 = box
    roi_patch = first_gray_img[y1:y2, x1:x2]

    # scale so the larger dimension hits target_size, regardless of the ROI's original shape
    h, w = roi_patch.shape
    scale = target_size / max(h, w)

    # upscale the actual image (not just the window) so click coordinates stay accurate
    display_img = cv2.resize(roi_patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    print(f"Sketch window: {roi_patch.shape[1]}x{roi_patch.shape[0]} -> "
          f"displayed at {display_img.shape[1]}x{display_img.shape[0]} (scale={scale:.2f})")

    drawing = {"active": False}
    sketch_points = []

    window_name = "Sketch around the slurry, then press any key when done"

    def draw_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["active"] = True
            sketch_points.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing["active"]:
                sketch_points.append((x, y))
                cv2.circle(display_img, (x, y), 2, (0, 255, 0), -1)
                cv2.imshow(window_name, display_img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing["active"] = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_event)
    cv2.imshow(window_name, display_img)

    # Use a polling loop instead of a single blocking waitKey(0) --
    # this keeps the GUI event loop cycling so mouse-driven redraws
    # (the green sketch line) actually render as you drag.
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key != 255:  # 255 means "no key pressed" during this poll
            break

    cv2.destroyAllWindows()

    if len(sketch_points) < 3:
        raise ValueError("Not enough points sketched to form a shape. Try again.")

    # scale points back down to the ORIGINAL image's pixel coordinates
    points_array = np.array(sketch_points, dtype=np.float32) / scale
    points_array = points_array.astype(np.int32)

    fixed_mask = np.zeros(roi_patch.shape, dtype="uint8")  # original (non-upscaled) size
    cv2.fillPoly(fixed_mask, [points_array], 255)

    Path("output").mkdir(exist_ok=True)
    cv2.imwrite(save_path, fixed_mask)
    print(f"Sketch-based mask saved to {save_path} -- open it to check the shape.")

    return fixed_mask


# ---------- Apply the fixed mask to a frame ----------
def analyze_frame(gray_img, color_img, box, fixed_mask, filename):
    """
    Applies the fixed mask (locked shape from frame 1) to the current frame.

    Returns:
      - mean_brightness: grayscale mean within the masked region right now
      - ice_fraction: what % of that fixed region still reads as bright/ice
      - mean_rgb: (R, G, B) mean color within the masked region right now

    Also saves two visual sanity-check images per frame:
      - output/masked_<filename>       : original colors, non-slurry blacked out
      - output/mask_overlay_<filename> : green outline of the fixed mask over the frame
    """
    x1, y1, x2, y2 = box
    gray_patch = gray_img[y1:y2, x1:x2]
    color_patch = color_img[y1:y2, x1:x2]

    # --- grayscale metrics (unchanged) ---
    region_pixels = gray_patch[fixed_mask == 255]
    mean_brightness = region_pixels.mean()

    _, current_threshold = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    still_ice = cv2.bitwise_and(current_threshold, fixed_mask)
    ice_fraction = still_ice[still_ice == 255].size / fixed_mask[fixed_mask == 255].size

    # --- NEW: RGB mean within the same fixed mask ---
    masked_color_pixels = color_patch[fixed_mask == 255]  # BGR order (OpenCV default)
    mean_bgr = masked_color_pixels.mean(axis=0)
    mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])  # reorder to R, G, B for readability

    # --- NEW: save a visual sanity-check image for this frame ---
    Path("output").mkdir(exist_ok=True)

    masked_visual = color_patch.copy()
    masked_visual[fixed_mask == 0] = 0  # black out everything outside the fixed slurry shape
    cv2.imwrite(f"output/masked_{filename}", masked_visual)

    overlay = color_patch.copy()
    contours, _ = cv2.findContours(fixed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # green outline
    cv2.imwrite(f"output/mask_overlay_{filename}", overlay)

    return mean_brightness, ice_fraction, mean_rgb


if __name__ == "__main__":
    # --- Load and convert ---
    images = load_images()
    print(f"\nLoaded {len(images)} image(s):")
    for filename, img in images:
        print(f"  {filename} -> shape {img.shape}")

    gray_images = convert_to_grayscale(images)
    print(f"\nConverted {len(gray_images)} image(s) to grayscale.")

    # --- Rim check (lighting stability) ---
    rim_petri_dish = (370, 947, 380, 965)
    print("\nRim brightness (lighting stability check):")
    for filename, gray_img in gray_images:
        brightness = get_brightness(gray_img, rim_petri_dish)
        print(f"  {filename}: rim brightness = {brightness:.2f}")

    ice_roi = (435 - 50, 1001 - 50, 906 + 50, 1382 + 50)
    time_interval = 15
    times = [i * time_interval for i in range(len(gray_images))]

    # --- Method 1: plain box ---
    print("\n[Box method] Ice brightness:")
    results = []
    for i, (filename, gray_img) in enumerate(gray_images):
        ice_brightness = get_brightness(gray_img, ice_roi)
        time_elapsed = times[i]
        results.append((time_elapsed, ice_brightness))
        print(f"  t={time_elapsed}s | {filename}: ice brightness = {ice_brightness:.2f}")

    times_list, brightness_list = zip(*results)
    plt.plot(times_list, brightness_list, marker="o")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mean Grayscale Intensity (Ice ROI)")
    plt.title("Ice Slurry Melting Progression (Box Method)")
    plt.show()

    # --- Method 2: sketch-based fixed mask ---
    first_filename, first_gray_img = gray_images[0]
    print(f"\nOpening sketch window on {first_filename} -- "
          f"click and drag to trace the slurry outline, then press any key.")
    fixed_mask = sketch_mask(first_gray_img, ice_roi)

    print("\n[Fixed-mask method] Tracking brightness + ice fraction + color:")
    mask_results = []
    for i, (filename, gray_img) in enumerate(gray_images):
        _, color_img = images[i]  # matching original color image, same order as gray_images
        mean_brightness, ice_fraction, mean_rgb = analyze_frame(
            gray_img, color_img, ice_roi, fixed_mask, filename
        )
        time_elapsed = times[i]
        mask_results.append((time_elapsed, mean_brightness, ice_fraction, mean_rgb))
        print(f"  t={time_elapsed}s | {filename}: "
              f"brightness={mean_brightness:.2f}, ice_fraction={ice_fraction:.3f}, "
              f"RGB=({mean_rgb[0]:.1f}, {mean_rgb[1]:.1f}, {mean_rgb[2]:.1f})")

    print("\nSaved per-frame visual sanity checks to output/masked_*.png and output/mask_overlay_*.png")

    mask_times, mask_brightness, mask_ice_fraction, mask_rgb = zip(*mask_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(mask_times, mask_brightness, marker="o", color="tab:blue")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Mean Brightness (fixed mask region)")
    ax1.set_title("Brightness Over Time (Fixed Mask)")

    ax2.plot(mask_times, mask_ice_fraction, marker="o", color="tab:orange")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Ice Fraction (0-1)")
    ax2.set_title("Remaining Ice Fraction Over Time")

    plt.tight_layout()
    plt.savefig("output/melt_progression_fixed_mask.png")
    plt.show()