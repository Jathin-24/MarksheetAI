# Free deployment: Streamlit Community Cloud (no code changes needed)

1. Create a GitHub repo and push this folder. `.gitignore` already keeps sample photos, student names and outputs OUT of the repo. `models/mnist_mlp.joblib` (2.6 MB) MUST be committed.
2. Go to share.streamlit.io -> sign in with GitHub -> **Create app** -> pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings**: Python version **3.12**. Secrets box: `APP_PASSWORD = "choose-a-long-code"`
4. Deploy. Open the URL on the phone -> "Add to Home Screen".

Notes: free apps go to sleep when unused (tap "wake up"); storage is temporary, so download the class Excel before closing; photos are processed in memory.
Optional TrOCR/PyTorch is too heavy for the free tier - leave `requirements.txt` as is.
