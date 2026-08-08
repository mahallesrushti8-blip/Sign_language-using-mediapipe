# Custom Sign Language Recognizer (OpenCV + MediaPipe)
<img width="479" height="384" alt="image" src="https://github.com/user-attachments/assets/2db6973f-64c7-4524-b0bb-c5ded02ff755" />
<img width="494" height="386" alt="image" src="https://github.com/user-attachments/assets/0c2b9aba-d979-4221-bd4e-f4cb3adfa507" />



This project recognizes hand signs you define yourself, using your
webcam. It works in 3 steps, run in order:

## 0. Setup (run once)
This needs to run on **your own computer** (with a webcam) — not in
this chat — since it needs live camera access.

```bash
pip install -r requirements.txt
```

## 1. Collect training data — `python collect_data.py`
- First, open the file and edit the `GESTURES` list with the names
  of the signs you want to teach it, e.g.:
  ```python
  GESTURES = ["hello", "thanks", "yes"]
  ```
- Run the script. A webcam window opens showing your hand skeleton.
- Make a sign and press its number key (0, 1, 2...) repeatedly while
  holding/slightly varying the pose, to save ~150-300 samples per
  sign. Try different angles, distances, and hand positions in frame
  so the model generalizes instead of memorizing one exact pose.
- Press `q` when done. This creates `landmarks.csv`.

## 2. Train the model — `python train_model.py`
- Reads `landmarks.csv`, trains a Random Forest classifier, prints
  an accuracy report, and saves `model.pkl`.
- If accuracy looks low (a lot of confusion between two signs),
  go back and collect more/cleaner data for those signs — that's
  almost always the fix, not the model.

## 3. Run live recognition — `python recognize.py`
- Opens your webcam and shows the predicted sign + confidence in
  real time. Press `q` to quit.

## How it works (short version)
- MediaPipe Hands detects 21 landmark points on your hand per frame
  (fingertips, knuckles, wrist), each with (x, y, z) coordinates —
  63 numbers total.
- Those 63 numbers are the "features." Instead of learning from raw
  pixels, the model learns from this compact geometric description
  of the hand shape, which is faster to train and needs far less
  data.
- A Random Forest classifier learns which patterns of coordinates
  correspond to which sign name, then predicts on new frames.

## Tips for better accuracy
- Keep your whole hand in frame and reasonably well lit.
- Record data with the SAME camera/setup you'll use for recognition.
- Add a "neutral"/"nothing" gesture class (e.g. relaxed hand) so it
  doesn't force a guess when you're not signing anything meaningful.
- This setup only handles static poses (one frame = one sign). If
  you eventually want signs that involve *movement* (e.g. waving),
  that needs a sequence model (like an LSTM) that looks at landmarks
  over several frames instead of just one — happy to help with that
  next if you get there.
