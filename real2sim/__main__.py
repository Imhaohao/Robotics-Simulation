import os

os.environ.setdefault("MUJOCO_GL", "egl")

from real2sim.pipeline import main

if __name__ == "__main__":
    main()
