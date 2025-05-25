#  Alzheimer’s Disease Diagnosis using Multimodal Deep Learning

This project implements a deep learning pipeline that fuses features from **MRI**, **PET**, and **DTI** imaging modalities to classify Alzheimer's Disease patients into:
- **AD** (Alzheimer’s Disease)
- **MCI** (Mild Cognitive Impairment)
- **CN** (Cognitively Normal)

The goal is to improve diagnostic accuracy using a **late fusion ensemble** of three separate CNN models trained on each modality.

---

##  Dataset Source

All imaging data was downloaded from the official **[ADNI (Alzheimer’s Disease Neuroimaging Initiative)](https://adni.loni.usc.edu/)** website.  
Datasets used:
- **MRI** (MPRAGE T1-weighted)
- **PET** (e.g., FDG-PET)
- **DTI** (Diffusion Tensor Imaging)

---

##  Project Structure

Alzheimer-Multimodal-Diagnosis/
│
├── MRI_Model.h5 # Trained CNN model for MRI
├── PET_Model.h5 # Trained CNN model for PET
├── DTI_Model.h5 # Trained CNN model for DTI
│
├── CIP_MRI/ # Preprocessed and split MRI dataset
│ └── split/test/
│
├── CIP_PET/ # Preprocessed and split PET dataset
│ └── split_pet_images/test/
│
├── CIP_DTI/ # Preprocessed and split DTI dataset
│ └── split_dti_images/test/
│
├── FusedCode.ipynb  # Script for fused model prediction
│


---

##  Evaluation

We used late fusion by combining the predictions of three individual CNN models. Fusion is done using **majority voting**.

 **Fused Model Accuracy on Test Set:**  
Achieved up to **84.58% (2436/2880)** accuracy.

> **Target:** Reach 95%+ by improving data augmentation, fine-tuning, or optimizing ensemble strategies.

---

##  Requirements

- Python 3.7+
- TensorFlow / Keras
- NumPy
- tqdm

 ## Install dependencies:
```bash
  pip install tensorflow keras numpy tqdm


 ## Future Improvements
  - Add data augmentation to increase generalization.
  - Apply 3D CNNs instead of 2D for volumetric learning.
  - Use attention-based fusion instead of majority voting.
  - Try transformer-based models (e.g., Swin Transformers).
