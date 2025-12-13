# Comparing Movie Recommendations Systems

Summary
-------
This project explores and compares multiple recommendation system approaches (collaborative filtering, content-based, hybrid) on a movie ratings dataset. The notebook shows data processing, model implementation, hyperparameter tuning, and evaluation using ranking and error metrics.

Skills Demonstrated
-------------------
- Recommender systems: collaborative filtering and content-based methods
- Data cleaning, feature engineering, and exploratory analysis
- Model evaluation with RMSE, top-k precision, and recall@k
- Use of `pandas`, `scikit-learn`, and recommender libraries (e.g., `scikit-surprise`)
- Comparative analysis and visualizations to justify model selection

Files
-----
- Comparing Movie Recommendations Systems.ipynb — primary Jupyter notebook
- Comparing Movie Recommendations Systems.pdf — exported report

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
```
pip install -r requirements.txt
```

Running the notebook
--------------------
Start Jupyter and open the notebook:

```bash
jupyter notebook "Comparing Movie Recommendations Systems.ipynb"
```

Notes on data
-------------
The notebook expects a movie ratings dataset (e.g., MovieLens). If the dataset is not included, follow the notebook instructions to download or place the data in this folder. Paths are relative to the project directory.

How to evaluate
------------------
- Focus on the evaluation section to see how different approaches compare on ranking metrics.
- The exploratory data analysis (EDA) cells highlight common data issues and how they were addressed.

License & Contact
-----------------
MIT-style: see root `LICENSE` for details. For questions, open an issue or contact the repository owner.
