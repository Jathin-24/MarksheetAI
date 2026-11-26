"""Fine-tune the bundled digit model with the teacher's own corrected digits (data/labelled/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2, numpy as np
from config import settings
from src import mnist_train

files = sorted(settings.LABELLED_DIR.glob("*.png"))
if not files:
    sys.exit("No corrected digits found in data/labelled/. Tick 'Save my corrections' on the Export screen first.")
X = np.array([cv2.imread(str(f), 0) for f in files], np.uint8)
y = np.array([int(f.name.split("_")[0]) for f in files])
print(f"retraining with {len(files)} corrected digits (weighted x20) ...")
mnist_train.train(extra=(X, y))
print("done: models/mnist_mlp.joblib updated")
