# How to Run

1. Open this folder in VS Code.
2. Create/activate a Python virtual environment.
3. Run `pip install -r requirements.txt`.
4. Run `python src/audit.py`.
5. Run `python src/eda.py`.
6. Run `python src/train.py`.
7. Put a new CSV/XLSX in `data/new/` and run `python src/test_new_dataset.py --input data/new/your_file.xlsx`.
8. Review `outputs/model_results/new_data_scored.csv`.

For the capstone, explain the leakage controls, why SVM was included, why Random Forest was selected, why the threshold is not automatically 0.50, and why customers served is separated from predictive features.
