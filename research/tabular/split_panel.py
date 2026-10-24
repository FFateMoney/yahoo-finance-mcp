"""Split data/panel.npz into two files under 30 MB each for copying to another machine."""
import numpy as np
from wf import HERE
z = np.load(f"{HERE}/data/panel.npz")
np.savez_compressed(f"{HERE}/data/panel_X.npz", X=z["X"])
np.savez_compressed(f"{HERE}/data/panel_meta.npz", **{k: z[k] for k in z.files if k != "X"})
