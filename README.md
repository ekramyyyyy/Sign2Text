# sign2text — ASL Citizen  Sign Recognition

A webcam-based American Sign Language (ASL) recognition system that detects **isolated signs** with
MediaPipe pose/hand landmarks, classifies them with a Transformer, and can optionally turn the
recognized gloss sequence into English and Arabic using an LLM.

```text
Webcam / video
      │
      ▼
MediaPipe Pose + Hands
      │
      ▼
67 landmarks × XYZ + frame-to-frame deltas
      │
      ▼
Pose Transformer
      │
      ▼
150 ASL Citizen glosses + confidence
      │
      ▼
 Groq / Gemini (optional)
      │
      ├──► English sentence
      └──► Arabic sentence
```

---

## 1. Current dataset and project scope

This version of the project uses **Microsoft ASL Citizen**, not WLASL.

The packaged model vocabulary contains **150 glosses**. The file
`web/model/labels.json` is the source of truth for the deployed class list.

ASL Citizen Version 1.0 contains 83,399 videos covering 2,731 signs from 52 participants.
Microsoft provides signer-independent train/validation/test splits. The original dataset is much
larger than the 150-gloss subset used by this project.

Official Microsoft resources:

- ASL Citizen project: https://www.microsoft.com/en-us/research/project/asl-citizen/
- Official download: https://www.microsoft.com/en-us/download/details.aspx?id=105253
- Dataset license: https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/
- Dataset datasheet: https://www.microsoft.com/en-us/research/project/asl-citizen/datasheet/
- Microsoft code/baselines: https://github.com/microsoft/ASL-citizen-code
- Paper: https://arxiv.org/abs/2304.05934

### Dataset migration note

The project **originally started with WLASL**. It was later migrated to **ASL Citizen**, so some
older WLASL-specific files and assumptions remained during the transition. This final version has
been cleaned up for the ASL Citizen workflow:

- ASL Citizen CSV metadata is used instead of WLASL JSON metadata.
- `Participant ID`, `Video file`, `Gloss`, and `ASL-LEX Code` are read from ASL Citizen metadata.
- The deployed vocabulary is 150 ASL Citizen glosses.

This note is intentional so the history of the dataset change is clear.

---

## 2. Important Microsoft dataset license notice

**ASL Citizen is not an MIT/Apache/GPL dataset and is not covered by this project's own source-code
terms. It is provided by Microsoft under the Microsoft Research License Terms.**

According to Microsoft's current ASL Citizen license:

- use is limited to **non-commercial, non-revenue-generating research purposes**, subject to the
  license;
- the ASL Citizen data and modifications of the data **may not be distributed**;
- Microsoft also restricts sharing, publishing, lending, or transferring the licensed Materials;
- personal data, if present, has additional handling and confidentiality requirements;
- results may be published provided that no material or substantial portion of the Materials is
  included.

Read the complete license before using, publishing, or redistributing this project:

**https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/**

This repository therefore does **not** include the 42.8 GB ASL Citizen video archive.

See [`DATASET-LICENSE.md`](DATASET-LICENSE.md) for the project-specific license notice.

---

## 3. What this project does

The system is designed for **isolated sign recognition** rather than unrestricted continuous ASL
translation.

For each sign segment, the pipeline:

1. captures webcam frames;
2. extracts body/hand landmarks with MediaPipe;
3. builds normalized pose features and temporal deltas;
4. feeds the sequence to a Transformer encoder;
5. predicts one of the 150 configured ASL Citizen glosses;
6. returns top-k predictions with confidence scores;
7. optionally sends recognized gloss tokens to an LLM;
8. displays an English and Arabic sentence.

The ASL Citizen creators specifically describe the dataset around isolated sign recognition and
dictionary-retrieval style applications. It should not be interpreted as a dataset for unrestricted
continuous-sign language understanding.

---

## 4. Repository structure

```text
sign2text/
├── asl_citizen.py              # ASL Citizen archive/CSV access
├── backup.py                   # Backup and restore generated data
├── check_parity.py             # Python/web preprocessing parity checks
├── config.py                   # Paths and model configuration
├── dataset.py                  # PyTorch ASL Citizen dataset
├── download_models.py          # Download MediaPipe model assets
├── export_onnx.py              # Export pose model to ONNX
├── extract_keypoints.py        # Download clips and extract landmarks
├── features.py                 # Feature utilities
├── keypoints.py                # MediaPipe landmark extraction
├── llm.py                      # Optional English/Arabic LLM conversion
├── model.py                    # SignNet / Transformer model
├── prepare_asl_citizen.py      # Build meta.csv + labels.json
├── preprocess.py               # Pose/RGB preprocessing
├── realtime.py                 # Local webcam inference
├── server.py                   # FastAPI local server
├── train.py                    # Model training/evaluation
├── video_io.py                 # Video loading utilities
├── requirements.txt
├── requirements-server.txt
├── DATASET-LICENSE.md
├── README.md
└── web/
    ├── app.js
    ├── features.js
    ├── index.html
    └── model/
        ├── hand_landmarker.task
        ├── labels.json
        ├── pose.onnx
        └── pose_landmarker_lite.task
```

---

## 5. Why the project uses a 150-gloss subset

ASL Citizen contains 2,731 glosses, while this project is configured for 150.

The 150-class setup is a practical vocabulary for training, debugging, real-time inference, and
deployment. It also keeps the model output space manageable while preserving the ASL Citizen data
format and signer-independent split structure.

The current deployed label file contains:

```text
150 unique ASL Citizen glosses
0-based class IDs
alphabetically ordered labels
```

To verify it:

```python
import json

labels = json.load(open("web/model/labels.json", encoding="utf-8"))

print("Number of glosses:", len(labels))
print("Unique glosses:", len(set(labels)))
print(labels[:10])
```

Expected:

```text
Number of glosses: 150
Unique glosses: 150
```

---

## 6. Colab workflow

The project is designed to avoid downloading the entire 42.8 GB ASL Citizen archive into a Colab
runtime.

`remotezip` can read the official ZIP through HTTP range requests. The workflow is:

```text
Official ASL Citizen ZIP
        │
        ├── remotely read train/val/test CSV files
        │
        ▼
select 150 glosses
        │
        ▼
download only required MP4 members
        │
        ▼
MediaPipe landmark extraction
        │
        ▼
.npy keypoint files
        │
        ▼
train pose Transformer
```

### Install

```bash
pip install -r requirements.txt
```

### Prepare the 150-gloss metadata

```bash
python prepare_asl_citizen.py --n 150
```

The command creates:

```text
data/meta.csv
data/labels.json
```

You can also provide a custom vocabulary:

```bash
python prepare_asl_citizen.py --glosses my_signs.txt
```

Or select another number of glosses for an experiment:

```bash
python prepare_asl_citizen.py --n 50
python prepare_asl_citizen.py --n 300
python prepare_asl_citizen.py --n 2731
```

The packaged project, however, is configured around **150 glosses**.

### Extract keypoints

```bash
python extract_keypoints.py
```

The extractor is resumable. Existing `.npy` files are skipped, and failed clips are recorded so
they can be investigated or retried.

---

## 7. Training

The default training path is the pose branch because it avoids repeatedly loading large video
files during training.

```bash
python train.py --mode pose --epochs 60 --bs 16
```

Optional inverse-frequency class weighting:

```bash
python train.py --mode pose --epochs 60 --bs 16 --weighted
```

The training script automatically gets the number of classes from:

```text
data/labels.json
```

For the packaged project this is 150.

Training reports:

- training set size;
- validation set size;
- test set size;
- number of classes;
- validation Top-1 accuracy;
- validation Top-5 accuracy;
- final test Top-1 accuracy;
- final test Top-5 accuracy.

The best validation checkpoint is saved under:

```text
checkpoints/
```

---

## 8. Model architecture

The default pose model uses:

```text
67 landmarks × 3 coordinates
        │
        ▼
201 coordinate features
        +
201 frame-to-frame delta features
        │
        ▼
402-dimensional frame representation
        │
        ▼
Linear projection
        │
        ▼
Transformer Encoder
        │
        ▼
CLS representation
        │
        ▼
Classification head
        │
        ▼
150 ASL Citizen classes
```

The landmark configuration contains:

- 25 pose landmarks;
- 21 left-hand landmarks;
- 21 right-hand landmarks.

---

## 9. Real-time inference

After training or restoring a compatible checkpoint:

```bash
python realtime.py --ckpt checkpoints/pose.pt
```

Useful controls:

```text
q          quit
c          clear recognized tokens
space      translate now
enter      translate now
```

The realtime application segments sign activity, predicts glosses, collects the accepted tokens,
and can pass the sequence to the configured LLM.

### Important

The model was trained on **unmirrored** video. Do not horizontally mirror the webcam feed during
inference unless the preprocessing/model pipeline is changed consistently.

---

## 10. Local web application

The project also contains a small FastAPI service for serving the web interface and keeping LLM API
keys off the browser.

Run:

```bash
uvicorn server:app --port 8000
```

Then open:

```text
http://localhost:8000
```

The web model uses:

```text
web/model/pose.onnx
web/model/labels.json
```

The deployed `labels.json` contains the same 150-class vocabulary expected by the model.

---

## 11. ONNX and browser inference

The pose model can be exported for browser-side inference:

```bash
python export_onnx.py
```

The browser-side pipeline is designed to mirror the Python preprocessing path:

```text
MediaPipe landmarks
       ↓
feature normalization
       ↓
temporal deltas
       ↓
ONNX pose model
       ↓
150-class prediction
```

`check_parity.py` can be used to compare preprocessing behavior between the Python and web
implementations.

---

## 12. Backup and restore

Colab runtimes are temporary, so extracted keypoints and checkpoints should be backed up.

Create a backup:

```bash
python backup.py save
```

Restore a backup:

```bash
python backup.py restore /content/sign2text_backup.tar
```

This is especially useful because the raw ASL Citizen archive is not intended to be copied into
the project repository.

---

## 13. LLM translation layer

The recognition model predicts **glosses**, not fluent English or Arabic sentences.

The optional LLM layer receives the recognized gloss sequence and produces:

```text
ASL glosses
    ↓
LLM
    ├── English
    └── Arabic
```

This distinction matters: the classifier itself is not an English/Arabic translation model.

Configure the provider/API credentials according to the implementation in `llm.py`.

---

## 14. Dataset limitations

ASL Citizen is a research dataset for isolated sign language recognition. Important limitations
include:

- the project only uses a 150-gloss subset;
- a single gloss does not represent all possible uses or grammatical forms of a sign;
- the system is not a general continuous-ASL translator;
- performance depends on the quality of MediaPipe landmark extraction;
- webcam conditions can differ from the dataset;
- signer-independent evaluation is important when reporting recognition performance;
- the dataset should not be treated as representative of the entire ASL-using population.

Microsoft's recommended-use documentation discusses these limitations and specifically cautions
against using ASL Citizen as a direct continuous-sign tokenization dataset.

---

## 15. Citation

If you use ASL Citizen, cite the original paper:

```bibtex
@article{desai2023asl,
  title={ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author={Desai, Aashaka and Berger, Lauren and Minakov, Fyodor O. and Milan, Vanessa
          and Singh, Chinmay and Pumphrey, Kriston and Ladner, Richard E. and
          Daum{\'e} III, Hal and Lu, Alex X. and Caselli, Naomi and Bragg, Danielle},
  journal={arXiv preprint arXiv:2304.05934},
  year={2023}
}
```

Paper:

https://arxiv.org/abs/2304.05934

Microsoft Research project:

https://www.microsoft.com/en-us/research/project/asl-citizen/

---

## 16. License and attribution summary

### Project code

The source code in this repository is the project code developed for this sign-recognition
application. No claim is made that this project code changes or replaces the license governing
ASL Citizen.

### ASL Citizen data

ASL Citizen remains subject to Microsoft's **Microsoft Research License Terms**.

Do not:

- add the ASL Citizen video archive to this repository;
- redistribute ASL Citizen videos or modified versions of the dataset;
- treat the dataset as an open-source dataset under this project's source-code license;
- use the dataset outside the permissions granted by Microsoft's license.

Always consult the current Microsoft license before distributing a project that uses ASL Citizen.

---

## 17. Acknowledgement

This project uses **ASL Citizen**, created by Microsoft Research and collaborators, with videos
contributed by members of the Deaf and hard-of-hearing community.

The dataset creators and contributors should receive appropriate attribution when the dataset is
used.

Official license:

https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/
