# Recording protocol (iPhone 15 Plus)

Target task: **LIBERO `libero_goal` task 8 — "put the bowl on the plate"**.

The iPhone 15 Plus has no LiDAR, so depth comes from a monocular metric-depth model
and the scale is fixed using the plate's measured diameter.

## Before recording

- [ ] Measure your **plate diameter** and **bowl rim diameter** in cm and write them below.
- [ ] Use a plain, matte table surface with nothing else on it.
- [ ] Use bright, even light (daylight or overhead light, no hard shadows).
- [ ] Remove watches and rings, and roll up your sleeve. Wear no gloves.

| Object | Diameter (cm) |
|---|---|
| Plate | |
| Bowl (rim) | |

## Camera setup

- Put the phone on a **tripod** or a stable stack of books. **It must not move** during a clip.
- Match the LIBERO camera: about **60–80 cm from the table**, looking down at **about 45°**, in front of the workspace.
- Use the **rear 1x camera**, **1080p at 30 fps**. Turn off Action mode, Cinematic mode and HDR video.
- **Lock exposure and focus**: long-press on the table in the Camera app until "AE/AF LOCK" appears.
- Hold the phone in landscape.
- Take **one photo** of the empty table with the plate on it, for calibration.

## Each clip (aim for 40, at least 30)

1. Start with your hand resting flat at the right edge of the frame and the bowl and plate apart.
2. Start recording, then wait about 1 second without moving.
3. Reach, grasp the bowl **by the rim with thumb and finger** (like a gripper pinch, not a full palm grab), lift, place it on the plate, release.
4. Return your hand to rest, wait about 1 second, then stop recording.

Each clip should be **3–8 seconds**.

**Vary between clips:** where the bowl starts and where the plate starts (about ±15 cm), and the approach speed a little.
**Keep fixed:** the camera, the lighting and the hand you use.

**Avoid:** covering the bowl with your whole hand, moving your hand out of frame, and fast jerky motions (they cause motion blur).

## Transfer

AirDrop or cable the videos into `data/raw/` and keep the original `IMG_xxxx.MOV` names.
Put the calibration photo at `data/raw/calib.jpg`.
