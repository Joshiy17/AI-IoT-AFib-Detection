# Integration of AI & IoT for Early Detection of Atrial Fibrillation (AFib)

**Project Student:** Yash Joshi  
**Student Number:** 2016AB001096  
**University:** Symbiosis University of Applied Sciences

**Project Description:**  
This project demonstrates how deep learning (CNN) can be used for early detection of Atrial Fibrillation (AFib) using ECG signals collected from smartwatch sensors. The dataset consists of 124 patients (84 normal, 40 AFib). The workflow includes simple signal smoothing and normalization, creating short sequences for training, and using a very basic convolutional neural network to classify ECG segments as AFib or normal.  
*This code is meant for academic and demonstration purposes only.*

## How to use

1. Make sure your working directory has:
    - `main.py`  
    - `inference.py`  
    - `data/ecg_data.csv` (with patient ECG samples)
    - `results/` folder for output model/results

2. Install requirements with:
   ```
   pip install numpy pandas scikit-learn tensorflow
   ```

3. To train and evaluate the model, run:
   ```
   python main.py
   ```

4. To test inference on a new ECG sample, run:
   ```
   python inference.py
   ```

## File overview

- **main.py:** Loads and cleans ECG data, trains a simple CNN, evaluates performance
- **inference.py:** Loads the trained model and predicts AFib probability on an ECG segment
- **data/ecg_data.csv:** ECG data for 124 patients, each with 40 readings
- **results/:** Saved trained model and metrics after training

- <img width="786" height="627" alt="image" src="https://github.com/user-attachments/assets/78df1484-a0c5-4c6a-9e4a-33d374c7f2ee" />


---
**This project was completed for academic purposes at Symbiosis University of Applied Sciences.**
