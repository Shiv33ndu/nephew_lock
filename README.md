# Nephew-Lock: Smart Device Privacy for Families
*Because my 9-month-old nephew shouldn't be emailing my boss.*

An intelligent screen-locking service designed for tablets and mobile devices. Using a Siamese Network architecture and MobileFaceNet, the application detects a specific child’s face and immediately locks the device to prevent unauthorized access or accidental "toddler-interference." Features a "Supervised Mode" that keeps the device unlocked when a recognized adult is also present in the frame.

This project utilizes MobileFaceNet to generate 128-d embeddings, optimized for resource-constrained Android devices. Key features include custom Triplet Loss training with semi-hard negative mining and an evolving "Gold Standard" embedding using Exponential Moving Averages (EMA) to account for facial changes as a child grows.

## Key Features

- **Privacy-First (Edge AI)**: No facial data is sent to the cloud. All inference and embedding generation happen locally on the device.
- **Intelligent Supervision Logic**: Features a "Dual-Face" logic gate that keeps the device unlocked if a recognized adult is also detected in the frame.
- **Adaptive Recognition**: Implements Exponential Moving Average (EMA) for embeddings, allowing the model to evolve as the child’s facial features change over time.
- **Low-Latency Performance**: Built on MobileFaceNet, optimized for high-speed inference on resource-constrained mobile hardware.

```
nephew_lock/
├── data/
│   ├── raw/                # Original 1:1 photos from your phone
│   │   ├── nephew/         # 40+ photos
│   │   ├── supervision/    # 60+ photos (Nephew + Adult)
│   │   └── negatives/      # Generic adult faces (LFW dataset)
│   ├── processed/          # Automated output from your cleaning scripts
│   │   ├── cropped_faces/  # 112x112 tight crops
│   │   └── embeddings/     # Pre-calculated .npy vectors for "Gold Standard"
├── src/
│   ├── data_prep/          # Scripts for cleaning and augmentation
│   │   ├── blur_detector.py
│   │   ├── face_cropper.py
│   │   └── align_faces.py
│   ├── models/             # Siamese Network architecture
│   │   ├── mobile_facenet.py
│   │   └── siamese_base.py
│   ├── training/           # Training loops and loss functions
│   │   └── train_triplet.py
│   └── inference/          # Deployment-ready code
│       ├── detector.py     # Real-time face detection logic
│       └── lock_trigger.py # Logic for locking the screen
├── notebooks/              # For Andrew Ng-style experimentation
│   ├── 01_data_exploration.ipynb
│   └── 02_threshold_tuning.ipynb
├── models_checkpoints/     # Saved .h5 or TFLite files
├── pyproject.toml          # Enterprise dependency locking
└── config.yaml             # Thresholds (e.g., lock_dist: 0.6)
```