"""One-time setup: (1) TrOCR weights, (2) torch CNN on MNIST. Needs internet once. Both are optional boosts."""
import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
settings.ensure_dirs()

print("== 1/2 TrOCR (microsoft/trocr-small-handwritten, ~250 MB) ==")
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TrOCRProcessor.from_pretrained(settings.TROCR_MODEL)
    VisionEncoderDecoderModel.from_pretrained(settings.TROCR_MODEL)
    print("TrOCR ready (cached by Hugging Face).")
except Exception as exc:
    print("TrOCR skipped:", exc)

print("== 2/2 CNN digit classifier ==")
try:
    import numpy as np, torch, torch.nn as nn
    from src.cnn_model import DigitCNN
    from src.mnist_train import augment, load_mnist
    X, y = load_mnist(per_class=3000)
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for img, lab in zip(X, y):
        xs.append(img); ys.append(lab)
        for _ in range(2):
            xs.append(augment(img, rng)); ys.append(lab)
    Xt = torch.tensor(np.array(xs, np.float32) / 255.0)[:, None]
    yt = torch.tensor(np.array(ys))
    net = DigitCNN(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    n = len(Xt); n_val = n // 20
    perm = torch.randperm(n); val, tr = perm[:n_val], perm[n_val:]
    for epoch in range(6):
        net.train(); p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), 128):
            b = p[i:i + 128]
            opt.zero_grad(); loss = nn.functional.cross_entropy(net(Xt[b]), yt[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(Xt[val]).argmax(1) == yt[val]).float().mean().item()
        print(f"epoch {epoch + 1}/6 validation accuracy {acc:.3f}")
    torch.save(net.state_dict(), settings.MODEL_DIR / "digit_cnn.pt")
    print("saved models/digit_cnn.pt")
except Exception as exc:
    print("CNN skipped:", exc)
