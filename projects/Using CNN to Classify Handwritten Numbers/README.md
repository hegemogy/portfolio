# Using CNN to Classify Handwritten Numbers

Summary
-------
This project trains a Convolutional Neural Network (CNN) to classify handwritten digits (MNIST-style). The notebook walks through data loading, preprocessing, model design with Keras/TensorFlow, training with validation, and model evaluation and visualization.

Skills Demonstrated
-------------------
- Deep learning: CNN architecture design and tuning
- Image preprocessing and augmentation
- Model training, validation, and evaluation (accuracy, loss curves)
- Use of TensorFlow / Keras and `matplotlib` for visualization
- Reproducible notebook workflow and model persistence

Files
-----
- Using CNN to Classify Handwritten Numbers.ipynb — primary Jupyter notebook
- Using CNN to Classify Handwritten Numbers.pdf — exported report

Getting started
---------------
Prerequisites

- Python 3.8 or newer
- `pip` available

Recommended: create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies

```bash
pip install -r requirements.txt
```

Running the notebook
--------------------
Start Jupyter and open the notebook:

```bash
jupyter notebook "Using CNN to Classify Handwritten Numbers.ipynb"
```

Notes on data
-------------
The notebook downloads and uses the MNIST dataset (via Keras datasets). If your environment blocks automatic downloads, place the dataset in the project directory and update the notebook data path accordingly.

How to evaluate
------------------
- See the training/validation plots and final test accuracy to evaluate model performance.
- Review the `Model` cell to examin architecture choices and hyperparameters.

License & Contact
-----------------
MIT-style: see root `LICENSE` for details. For questions, open an issue or contact the repository owner.
