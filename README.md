# Ice Slurry Melt Analysis

A Python/OpenCV pipeline for quantifying how ice slurry melts over time from a sequence of photos. It tracks two things per frame: **mean grayscale brightness** of the ice region, and **ice fraction** (how much of a fixed container area is still ice).

## Motivation

Judging melt progress by eye is subjective. This project turns a photo sequence into numbers that can be plotted against time. Brightness is used as a practical proxy for opacity: opaque ice scatters light and reads bright, while clear meltwater is more transparent and reads darker.

## How it works

1. **Load** PNG frames from `images/`, sorted by filename.
2. **Convert** each frame to grayscale.
3. **Check lighting stability** by measuring brightness of a small, static reference patch (the petri dish rim) in every frame. If the reference stays roughly constant, changes in the ice region reflect melting rather than lighting drift.
4. **Box method:** mean grayscale intensity of a fixed rectangular ROI over time. Simple and robust, but the box includes some non-ice area.
5. **Sketch method (two-tier manual masking):**
   - **Step A, container region (once, on frame 1):** you trace the overall area where slurry and liquid sit. Its pixel count is the fixed denominator for every frame.
   - **Step B, ice region (every frame, including frame 1):** you trace just the ice within that container. Its pixel count is that frame's numerator.
   - `ice_fraction = ice pixels sketched in this frame / fixed container pixels`

   Because the denominator is fixed, frame 1 does not have to equal 1.0. If liquid was already present at the start (for example, it came out of the syringe with the ice), that shows up in the numbers.
6. **Plot** brightness and ice fraction against elapsed time.

## Why manual sketching?

Automatic segmentation was tried first and did not hold up on this data:

| Approach | Problem |
|---|---|
| Otsu thresholding | Glare on wet surfaces and bright background read as "ice"; ice fraction was inflated in late, mostly-liquid frames |
| HSV color segmentation | Ice, liquid, and background fell in nearly the same hue/saturation range, so there was no color contrast to threshold on |
| Largest connected blob | Assumes the ice stays one bright connected region; breaks as it melts, and touching glare streaks merge into the ice blob |
| Erosion to split merged blobs | Too aggressive on real ice edges, not aggressive enough on thick bridges |

Hand-tracing the ice boundary is slower but unambiguous, and it keeps the measurement tied to what a person actually sees as ice.

## Project structure

```
ice-slurry-cv/
├── images/          # input photos (not tracked by git)
├── src/
│   └── load_images.py   # full pipeline
├── output/          # generated masks, visual checks, plots
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/ice-slurry-cv.git
cd ice-slurry-cv

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Core dependencies: `opencv-python`, `numpy`, `matplotlib`. Use the standard `opencv-python` package, not `opencv-python-headless`, because the sketching tool needs OpenCV's GUI windows.

## Usage

1. Put your PNG frames in `images/`, named so they sort in time order (e.g. `ice_slurry1.png`, `ice_slurry2.png`, ...). Use zero-padding (`01`, `02`, ...) if you have 10 or more frames so alphabetical order stays chronological.
2. Edit the constants near the bottom of `src/load_images.py`:
   - `rim_petri_dish`: `(x1, y1, x2, y2)` box on a small patch of static rim, used for the lighting check
   - `ice_roi`: `(x1, y1, x2, y2)` box containing the whole slurry in every frame
   - `time_interval`: seconds between photos
3. Run from the project root:

```bash
python src/load_images.py
```

### What to expect when it runs

1. Printed rim brightness for each frame (should be roughly constant).
2. A plot window for the box method. **Close it to continue.**
3. **Step A:** one sketch window on frame 1. Click and drag to trace the full container region, then press any key.
4. **Step B:** one sketch window per frame. Trace only the ice (not clear liquid or glare), then press any key.
5. A final two-panel plot (brightness and ice fraction vs. time).

Coordinates are in `(x1, y1, x2, y2)` order, with `(x1, y1)` the top-left and `(x2, y2)` the bottom-right corner. When slicing arrays, NumPy indexes `[row, column]`, so the code uses `img[y1:y2, x1:x2]`.

## Outputs

All written to `output/`:

| File | Description |
|---|---|
| `container_mask.png` | Fixed container region traced on frame 1 |
| `ice_mask_<frame>.png` | Ice region traced for each frame |
| `masked_<frame>.png` | Original color frame with everything outside the ice mask blacked out, for visual sanity checks |
| `melt_progression_sketch.png` | Brightness and ice fraction vs. time |

Check the mask images before trusting the numbers.

## Assumptions and limitations

- **Fixed camera and dish position.** The ROI and rim patch use the same pixel coordinates in every frame. If the camera or dish moves, frames need to be aligned first.
- **Stable lighting.** The rim check is a sanity check, not a correction. It does not normalize frames.
- **Brightness is a proxy for opacity.** Glare or reflections off wet surfaces can look as bright as ice, which is why the ice region is traced by hand.
- **Manual tracing is subjective.** Different people (or the same person on different days) will trace slightly different boundaries. Treat small changes in ice fraction between frames with caution.
- **Small sample sizes.** With only a handful of frames, trends are suggestive rather than statistically strong. More frames over a longer, more visible melt give a more trustworthy curve.
- **Hard-coded coordinates.** ROI and rim boxes are set in the script rather than selected interactively.
- **No melt-rate fit yet.** The script produces the time series and plots but does not fit a curve or report a rate.

## Possible next steps

- Fit a linear or exponential model to the ice-fraction curve to report a melt rate.
- Select the ROI and rim patch interactively instead of hard-coding them.
- Reduce glare at capture time (diffused lighting, slight camera angle) so automatic segmentation becomes viable.
- Try a learned segmentation approach once there are enough labeled frames (the hand-traced masks could serve as training labels).
- Read EXIF timestamps instead of assuming a fixed interval.

## Author
Developed as an independent Biomedical Engineering and computer vision project.

**Key Areas**: Computer Vision | Image Processing | Python | Data Analysis
