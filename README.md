# Human Video → Robot Policy

Driving a simulated robot with manipulation data I recorded myself on an iPhone.

**Target task:** LIBERO `libero_goal` #8, "put the bowl on the plate" (Franka Panda).

## Pipeline (in progress)

```
iPhone video ─► hand pose + metric depth ─► wrist 6-DoF + grasp aperture
            ─► retarget to Panda EE ─► replay in LIBERO (keep successes)
            ─► LeRobot dataset ─► fine-tune SmolVLA ─► evaluate
```

## Setup

```bash
./scripts/setup.sh          # Python 3.12 venv, LeRobot + SmolVLA + LIBERO
./scripts/smoke_eval.sh     # pretrained SmolVLA baseline on the target task
```

Tested on an RTX 4090 Laptop GPU (16 GB), Ubuntu, Python 3.12.

## Layout

| Path | Contents |
|---|---|
| `RECORDING.md` | Data collection protocol |
| `src/extract/` | Hand, depth and object extraction from video |
| `src/retarget/` | Human hand → robot end-effector mapping |
| `src/sim/` | LIBERO replay and dataset export |
| `scripts/` | Entry points |
| `results/` | Metrics and videos |
