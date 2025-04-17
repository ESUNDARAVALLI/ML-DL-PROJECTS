
TITLE: Deep Learning-Based Environmental Risk Assessment for Household E-Waste

This project is a machine learning pipeline that uses a deep learning model built with PyTorch to classify household e-waste environmental risk levels.

The model processes household e-waste data, applies label encoding, one-hot encoding for categorical features, feature scaling, and finally trains a neural network to predict the `Risk_Level` of waste data.


Dataset:

The dataset used is named: HOUSEHOLD_E-WASTE_ENV_RISK.csv

Make sure the CSV file is placed in the same directory as your notebook or script.

Features:

- Preprocessing: Label Encoding, One-Hot Encoding, Standard Scaling.
- Model: A fully connected deep learning network using PyTorch.
- Loss: CrossEntropyLoss.
- Optimizer: Adam.
- Output: Classification of risk level into three classes.

Usage

Install Dependencies:

```bash
pip install torch torchvision scikit-learn pandas
```

---

Run the Notebook:

If you're using Google Colab or Jupyter:

1. Upload the `HOUSEHOLD_E-WASTE_ENV_RISK.csv` file.
2. Copy-paste or run the provided code cells.
3. The model will train for 50 epochs and print the training loss.


Evaluate:

After training, the model calculates the prediction accuracy on the test dataset:

PyTorch DL Accuracy:

              precision    recall  f1-score   support

        High       0.80      1.00      0.89        16
         Low       0.91      1.00      0.95        21
    Moderate       1.00      0.93      0.96        86

    accuracy                           0.95       123
   macro avg       0.90      0.98      0.94       123
weighted avg       0.96      0.95      0.95       123


Confusion Matrix:
[[16  0  0]
 [ 0 21  0]
 [ 4  2 80]]



Project Structure:

.
├── your_model_notebook.ipynb
├── HOUSEHOLD_E-WASTE_ENV_RISK.csv
├── README.md

Model Architecture:

- Input Layer: Matches the number of processed features.
- Hidden Layer 1: 128 units with ReLU activation.
- Hidden Layer 2: 64 units with ReLU activation.
- Output Layer: 3 units (for three risk classes).

Author: ESUNDARAVALLI

License

This project is open-source and free to use for academic or research purposes.


You can copy this into a file named `README.md` in your repository.  
Want me to suggest a `.gitignore` file too? Let me know!
