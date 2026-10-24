"""Split data/panel.npz into three files under 25 MB each for copying to another machine."""
import numpy as np
from wf import HERE
z = np.load(f"{HERE}/data/panel.npz")
h = len(z["X"]) // 2
np.savez_compressed(f"{HERE}/data/panel_X1.npz", X=z["X"][:h])
np.savez_compressed(f"{HERE}/data/panel_X2.npz", X=z["X"][h:])
np.savez_compressed(f"{HERE}/data/panel_meta.npz", **{k: z[k] for k in z.files if k != "X"})
