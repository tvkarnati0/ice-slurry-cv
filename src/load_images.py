"""
Ice slurry melting analysis pipeline.

Two parallel methods are run:
  1. Box method    -- plain rectangular ROI mean intensity
  2. Sketch method -- two-tier manual sketching:
       a. FIXED CONTAINER REGION: sketched ONCE, on frame 1. Defines
          the overall area where slurry/liquid sits. This stays
          constant across all frames and is the ice_fraction
          DENOMINATOR throughout.
       b. PER-FRAME ICE SKETCH: sketched on EVERY frame (including
          frame 1), tracing just the actual ice within the container.
          This is the ice_fraction NUMERATOR for that frame.

     ice_fraction = (ice pixels sketched this frame) / (fixed container pixels)

     Because the denominator is fixed but the numerator is re-sketched
     each frame, frame 1 does NOT have to equal 1.0 -- if some liquid
     was already present at the start, frame 1's fraction reflects that.
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
def sketch_mask(gray_img, box, save_path, prompt_label, target_size=1200):
    """
    Opens a window showing this frame's ROI, upscaled so its largest
    dimension is roughly `target_size` pixels. Click and drag to sketch
    an outline. Press any key when done. Returns the filled-in binary
    mask, sized to match the ORIGINAL (non-upscaled) ROI dimensions.

    `prompt_label` is shown in the window title so you know what to
    sketch (e.g. "CONTAINER region" vs "ICE only").
    """
    x1, y1, x2, y2 = box
    roi_patch = gray_img[y1:y2, x1:x2]

    h, w = roi_patch.shape
    scale = target_size / max(h, w)

    display_img = cv2.resize(roi_patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    drawing = {"active": False}
    sketch_points = []

    window_name = f"Sketch the {prompt_label}, then press any key -- {save_path}"

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

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key != 255:
            break

    cv2.destroyAllWindows()

    if len(sketch_points) < 3:
        raise ValueError("Not enough points sketched to form a shape. Try again.")

    points_array = np.array(sketch_points, dtype=np.float32) / scale
    points_array = points_array.astype(np.int32)

    mask = np.zeros(roi_patch.shape, dtype="uint8")
    cv2.fillPoly(mask, [points_array], 255)

    Path("output").mkdir(exist_ok=True)
    cv2.imwrite(save_path, mask)

    return mask


# ---------- Analyze one frame: ice sketch relative to fixed container ----------
def analyze_frame(gray_img, color_img, box, ice_mask, container_pixel_count, filename):
    """
    Uses this frame's hand-sketched ice_mask, measured against the
    FIXED container pixel count (same for every frame), to compute:
      - mean_brightness: grayscale mean within the sketched ice region
      - ice_fraction: (ice pixels sketched this frame) / (fixed container pixels)
      - mean_rgb: (R, G, B) mean color within the sketched ice region

    Also saves a visual sanity-check image:
      - output/masked_<filename> : original colors, non-ice blacked out
    """
    x1, y1, x2, y2 = box
    gray_patch = gray_img[y1:y2, x1:x2]
    color_patch = color_img[y1:y2, x1:x2]

    region_pixels = gray_patch[ice_mask == 255]
    mean_brightness = region_pixels.mean()

    ice_fraction = np.sum(ice_mask == 255) / container_pixel_count

    masked_color_pixels = color_patch[ice_mask == 255]  # BGR order
    mean_bgr = masked_color_pixels.mean(axis=0)
    mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])

    Path("output").mkdir(exist_ok=True)
    masked_visual = color_patch.copy()
    masked_visual[ice_mask == 0] = 0
    cv2.imwrite(f"output/masked_{filename}", masked_visual)

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

    ice_roi = (435-100, 1001-100, 906+100, 1382+100)
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

    # --- Method 2, Step A: sketch the FIXED CONTAINER region, once, on frame 1 ---
    first_filename, first_gray_img = gray_images[0]
    print(f"\nStep A: sketch the CONTAINER region (overall slurry/liquid area) on {first_filename}.")
    print("This defines the fixed denominator used for every frame's ice_fraction.")
    container_mask = sketch_mask(
        first_gray_img, ice_roi,
        save_path="output/container_mask.png",
        prompt_label="CONTAINER region (overall slurry/liquid area)"
    )
    container_pixel_count = np.sum(container_mask == 255)
    print(f"Container area locked in: {container_pixel_count} pixels.")

    # --- Method 2, Step B: sketch the actual ICE on EVERY frame ---
    print("\nStep B: sketch just the ICE (not liquid) on each frame as it opens.")
    mask_results = []
    for i, (filename, gray_img) in enumerate(gray_images):
        _, color_img = images[i]

        print(f"  -> Sketching ICE in {filename} ({i + 1}/{len(gray_images)})...")
        ice_mask = sketch_mask(
            gray_img, ice_roi,
            save_path=f"output/ice_mask_{filename}",
            prompt_label="ICE only (not liquid)"
        )

        mean_brightness, ice_fraction, mean_rgb = analyze_frame(
            gray_img, color_img, ice_roi, ice_mask, container_pixel_count, filename
        )
        time_elapsed = times[i]
        mask_results.append((time_elapsed, mean_brightness, ice_fraction, mean_rgb))
        print(f"     t={time_elapsed}s | brightness={mean_brightness:.2f}, "
              f"ice_fraction={ice_fraction:.3f}, "
              f"RGB=({mean_rgb[0]:.1f}, {mean_rgb[1]:.1f}, {mean_rgb[2]:.1f})")

    print("\nSaved container mask to output/container_mask.png")
    print("Saved per-frame ice masks to output/ice_mask_*.png and visuals to output/masked_*.png")

    mask_times, mask_brightness, mask_ice_fraction, mask_rgb = zip(*mask_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(mask_times, mask_brightness, marker="o", color="tab:blue")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Mean Brightness (sketched ice region)")
    ax1.set_title("Brightness Over Time (Sketch Method)")

    ax2.plot(mask_times, mask_ice_fraction, marker="o", color="tab:orange")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Ice Fraction (relative to fixed container)")
    ax2.set_title("Remaining Ice Fraction Over Time")

    plt.tight_layout()
    plt.savefig("output/melt_progression_sketch.png")
    plt.show()