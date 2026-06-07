"""Generate synthetic sample episode data for testing the pipeline.

Creates a minimal episode directory with:
- front_rgb/ (synthetic images)
- wrist_rgb/ (synthetic images)
- meta.json
- state.json
- action.json
- gripper.json
- force_torque.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_sample_episode(output_dir: str | Path, num_frames: int = 30, fps: float = 30.0) -> None:
    """Generate a synthetic pick-and-place episode.

    Simulates:
    - Frames 0-9: reaching (gripper open, moving toward object)
    - Frames 10-14: grasping (gripper closing, force spike)
    - Frames 15-24: lifting (gripper closed, moving up)
    - Frames 25-29: placing (gripper opening, moving down)

    Args:
        output_dir: Where to write the episode directory.
        num_frames: Number of frames to generate.
    """
    output_dir = Path(output_dir)
    (output_dir / "front_rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "wrist_rgb").mkdir(parents=True, exist_ok=True)

    w, h = 640, 480

    # Object position (simulated)
    obj_x, obj_y = 320, 300  # Object center in image
    obj_r = 40  # Object radius

    states = []
    actions = []
    gripper_states = []
    force_torques = []

    for i in range(num_frames):
        t = i / num_frames  # normalized time [0, 1)

        # Simulate phases
        if i < 10:
            # Reach phase: arm moves toward object
            phase = "reach"
            progress = i / 10
            # Gripper is open
            gripper = 1.0
            # Robot moves from left toward object
            arm_x = 100 + (obj_x - 100) * progress
            arm_y = 350 - 50 * progress
            # Low force
            fx, fy, fz = 0.5, 0.5, 0.3
        elif i < 15:
            # Grasp phase: gripper closes
            phase = "grasp"
            progress = (i - 10) / 5
            gripper = 1.0 - progress * 0.8  # closing
            arm_x = obj_x
            arm_y = obj_y + 30
            # Force spike when gripping
            fx = 2.0 + 8.0 * progress
            fy = 1.0 + 3.0 * progress
            fz = 5.0 + 10.0 * progress
        elif i < 25:
            # Lift phase: arm moves up
            phase = "lift"
            progress = (i - 15) / 10
            gripper = 0.2
            arm_x = obj_x
            arm_y = obj_y + 30 - 80 * progress
            # Moderate force (holding object)
            fx, fy, fz = 6.0, 3.0, 12.0
        else:
            # Place phase: arm moves down, gripper opens
            phase = "place"
            progress = (i - 25) / 5
            gripper = 0.2 + progress * 0.8
            arm_x = obj_x + 50 * progress
            arm_y = obj_y - 50 + 80 * progress
            fx, fy, fz = 3.0, 1.5, 6.0

        # Generate front RGB image
        img = Image.new("RGB", (w, h), (40, 40, 50))
        draw = ImageDraw.Draw(img)

        # Draw table surface
        draw.rectangle([50, 350, 590, 450], fill=(80, 60, 40))

        # Draw target object (red circle)
        draw.ellipse(
            [obj_x - obj_r, obj_y - obj_r, obj_x + obj_r, obj_y + obj_r],
            fill=(200, 50, 50),
            outline=(255, 80, 80),
        )

        # Draw gripper (two parallel lines)
        grip_half = 15 + (1.0 - gripper) * 10
        draw.line(
            [(arm_x - grip_half, arm_y - 20), (arm_x - grip_half, arm_y + 20)],
            fill=(180, 180, 180), width=3,
        )
        draw.line(
            [(arm_x + grip_half, arm_y - 20), (arm_x + grip_half, arm_y + 20)],
            fill=(180, 180, 180), width=3,
        )
        # Gripper base
        draw.line(
            [(arm_x - grip_half, arm_y), (arm_x + grip_half, arm_y)],
            fill=(150, 150, 150), width=2,
        )

        # Draw arm
        draw.line([(arm_x, arm_y + 20), (arm_x, 100)], fill=(120, 120, 120), width=4)

        # Phase label
        draw.text((10, 10), f"Frame {i:03d} | {phase}", fill=(255, 255, 255))

        img.save(output_dir / "front_rgb" / f"{i:06d}.png")

        # Generate wrist RGB (simpler view)
        wrist_img = Image.new("RGB", (320, 240), (60, 60, 70))
        wrist_draw = ImageDraw.Draw(wrist_img)
        # Show close-up of gripper area
        wrist_draw.rectangle([100, 80, 220, 160], fill=(100, 80, 60))
        grip_w = int(30 + (1.0 - gripper) * 20)
        wrist_draw.rectangle([160 - grip_w, 100, 160 + grip_w, 140], fill=(180, 180, 180))
        wrist_img.save(output_dir / "wrist_rgb" / f"{i:06d}.png")

        # Record state (7-DOF: 6 joints + gripper)
        state = [
            arm_x / w,  # normalized x
            arm_y / h,  # normalized y
            0.5,  # z
            0.0, 0.0, 0.0,  # orientation
            gripper,
        ]
        states.append(state)

        # Action (velocity)
        if i > 0:
            action = [s - ps for s, ps in zip(state, states[-2])]
        else:
            action = [0.0] * 7
        actions.append(action)

        gripper_states.append(gripper)
        force_torques.append([fx, fy, fz, 0.1, 0.1, 0.05])

    # Save metadata
    meta = {
        "episode_id": "sample_episode_001",
        "task_instruction": "pick up the red cup from the table",
        "robot_name": "sim_robot_v1",
        "fps": fps,
        "camera_intrinsics": {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
        },
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(output_dir / "state.json", "w") as f:
        json.dump(states, f, indent=2)

    with open(output_dir / "action.json", "w") as f:
        json.dump(actions, f, indent=2)

    with open(output_dir / "gripper.json", "w") as f:
        json.dump(gripper_states, f, indent=2)

    with open(output_dir / "force_torque.json", "w") as f:
        json.dump(force_torques, f, indent=2)

    print(f"Generated {num_frames} frames at {output_dir}")
