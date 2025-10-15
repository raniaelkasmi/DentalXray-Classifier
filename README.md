# 🦷 DentalXray-Classifier  
*A deep learning project by Rania El Kasmi (Politecnico di Milano & EPFL)*  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/upload)  
*(Click the badge above to open the notebook directly in Google Colab)*  

---

## 🎯 Project Overview
This project demonstrates how convolutional neural networks (CNNs) can classify **dental panoramic X-ray images (OPGs)** into multiple diagnostic categories using **transfer learning (MobileNetV2)**.

It was developed as a **beginner-friendly deep learning pipeline** to explore computer vision applications in biomedical imaging, with a focus on dental radiography.

---

## 🧠 Dataset
The dataset used is the **Dental OPG X-Ray Dataset (Version 4)**, available on [Kaggle](https://www.kaggle.com/datasets/orvillejain/dental-opg-xray-dataset) under a CC BY 4.0 License.

### 📂 Classes (6 categories)
Each sub-folder corresponds to a diagnostic label:
- 🦷 `BDC-BDR` – Bone disease or bone resorption  
- ⚫ `Caries` – Dental caries  
- 💔 `Fractured_Teeth` – Tooth fracture  
- 💡 `Healthy_Teeth` – Normal teeth  
- 📉 `Impacted_Teeth` – Impacted or unerupted teeth  
- 🧫 `Infection` – Signs of periapical infection  

The dataset was organized as follows:
```
MyDrive/DentalXray-Classifier/data/
├── BDC-BDR/
├── Caries/
├── Fractured_Teeth/
├── Healthy_Teeth/
├── Impacted_Teeth/
└── Infection/
```

---

## ⚙️ Model Architecture
- **Base model:** MobileNetV2 (pre-trained on ImageNet)  
- **Input size:** 224 × 224 × 3 pixels  
- **Fine-tuning:** Top layers trained for 8 epochs (frozen base) + 5 epochs (unfrozen)  
- **Optimizer:** Adam  
- **Loss:** Sparse Categorical Cross-Entropy  
- **Metrics:** Accuracy  

---

## 🚀 Results (example)
| Metric | Training | Validation |
|:--|:--:|:--:|
| Accuracy | 0.95 | 0.87 |
| Loss | 0.18 | 0.33 |

*(Values will vary slightly depending on random seed and data split.)*

### 📊 Confusion Matrix  
Displays per-class performance and confusion patterns.  

### 🔥 Grad-CAM Visualization  
Highlights the image regions most influential in the model’s prediction.

---

## 💻 How to Run on Google Colab
1. Upload the dataset folder `data/` into your Drive:  
   `MyDrive/DentalXray-Classifier/data/`
2. Open the notebook:  
   [DentalXray_Classifier.ipynb](https://colab.research.google.com/upload)
3. Run all cells in order:
   - Mount Drive  
   - Load dataset  
   - Train model  
   - Visualize results  
   - Save model (`dental_classifier_mobilenetv2.h5`)

---

## 📦 Requirements
```
tensorflow>=2.14
numpy
pandas
matplotlib
scikit-learn
opencv-python
```

---

## 🧩 Repository Structure
```
DentalXray-Classifier/
├── data/                 # dataset (not pushed to GitHub)
├── notebooks/
│   └── DentalXray_Classifier.ipynb
├── images/               # figures (confusion matrix, Grad-CAM)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📜 License & Ethics
- Dataset © Orville Jain 2024, CC BY 4.0 License  
- This project is for **educational and research purposes only**  
  → *Not a clinical diagnostic tool.*

---

## 👩‍💻 Author
**Rania El Kasmi**  
MSc Student — Politecnico di Milano / Exchange at EPFL  
📧 r.elkasmi@polimi.it  
📍 Milan, Italy  

---

*“AI won’t replace dentists — but dentists using AI will replace those who don’t.”*
