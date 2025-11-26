
# 🚦 Traffic Sign Classification  
### 🧠 Computer Vision Project • 📷 Classical ML + CNNs • 📊 Comparative Evaluation

This project implements a complete **traffic sign recognition pipeline** using both **classical computer vision** and **deep learning** techniques.  
The goal is to compare how feature-engineering approaches perform against modern CNN architectures on the same dataset.

Dataset used:  
📁 **Kaggle Traffic Sign Dataset**  
🔗 https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification

---

## 🚀 Features

- 🧩 **Multiple Classification Pipelines Implemented**
  - HOG + SVM  
  - SIFT Bag-of-Visual-Words (BoVW) + SVM  
  - Convolutional Neural Network (CNN) classifier  

- 🔧 **Full data preprocessing**  
  - Image resizing  
  - Normalization  
  - Train/Validation split  
  - Data augmentation  

- 📊 **Performance Benchmarking** across:
  - Accuracy  
  - Robustness  
  - Generalization to unseen sign categories  

- 🖼️ **Visualizations & Analysis**
  - HOG & SIFT descriptor visualizations  
  - Confusion matrices  
  - Prediction displays  
  - Model comparison graphs  

- 📓 **Well-organized Jupyter Notebooks**  
  - Each pipeline has separate training & analysis notebooks  

---

## 🧠 Implemented Models

### 🔹 HOG + SVM  
- Extracted Histogram of Oriented Gradients features  
- Trained a Support Vector Machine classifier  
- Strong baseline performance for simple signs  

### 🔹 SIFT BoVW + SVM  
- Extracted SIFT keypoints  
- Built a Bag-of-Visual-Words dictionary  
- Trained SVM on histogram-of-visual-words representations  
- Improved generalization for textured signs  

### 🔹 CNN Classifier  
- Custom lightweight CNN architecture  
- Used data augmentation for robustness  
- Achieved the highest accuracy on the dataset  

---

## 📦 Installation

### 1️⃣ Create a virtual environment
```bash
python3 -m venv traffic_env
````

Activate:

```bash
# Windows
traffic_env\Scripts\activate

# Mac/Linux
source traffic_env/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 📓 Jupyter Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open the notebooks:

* `HOG_SVM.ipynb`
* `SIFT_BoVW_SVM.ipynb`
* `CNN_Classifier.ipynb`

---

## 📊 Results & Evaluation

* 📌 Classical ML pipelines (HOG, SIFT) achieved **solid accuracy** for simple categories
* 📌 SIFT BoVW outperformed HOG for detailed signs due to richer feature descriptors
* 📌 CNN achieved **the highest accuracy overall**, showing stronger generalization
* 📉 Confusion matrices reveal common misclassifications and dataset ambiguities
* 📈 Visualizations provided insights into feature quality and decision boundaries

---

## 🛠️ Tech Stack

* Python
* OpenCV
* Scikit-learn
* TensorFlow / Keras or PyTorch
* NumPy, Matplotlib, Seaborn
* Jupyter Notebook

---

## 🔮 Future Enhancements

* Add transfer learning (ResNet, MobileNet)
* Implement real-time traffic sign detection using OpenCV
* Add noise/blur robustness tests
* Deploy CNN model as a Streamlit web app

---


## 🙌 Credits

Developed by **Esha** 💛
Dataset Credit: Kaggle — A. Hema Teja


