# Iris Species Classification: Scaling, Neighbors, and Validation

This portfolio project asks whether petal length and petal width can classify an Iris flower as **setosa**, **versicolor**, or **virginica**. It demonstrates how to standardize measurements using only eligible training data, choose the number of neighbors (`k`) on a validation set, and evaluate the frozen model once on unseen test data. The six flowers in the hand-calculation section of [`memo.md`](memo.md) are invented teaching examples; the Python experiment uses all 150 real Iris observations bundled with scikit-learn.

The two inputs are **petal length (cm)** and **petal width (cm)**, in that order. The original species codes are `0 = setosa`, `1 = versicolor`, and `2 = virginica`. The species code and dataset row number are never distance features. No dataset download is needed after installing scikit-learn. The [scikit-learn Iris documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) describes the bundled dataset, which originates from Fisher's Iris data.

## Method and findings

The script splits the 150 row indices into 120 development and 30 test rows, then splits development into 90 training and 30 validation rows. Both splits are stratified with `random_state=42`. It fits a fresh `StandardScaler` → Euclidean, uniform-vote KNN pipeline on training rows for each `k` in `1, 3, 5, 7, 9, 15, 25`. The highest validation accuracy wins; an exact tie goes to the smaller `k`. After choosing `k`, a **new** pipeline is fitted on all 120 development rows. A most-frequent-class baseline is fitted on those same development rows. Only then are both models evaluated on the 30 held-out test rows.

With the tested versions below, `k = 3` wins: `k = 3, 5, 7, 9, 15, 25` each score `0.933333` on validation, so the smaller `k` breaks the tie. On test rows, KNN accuracy is `0.966667` and macro-F1 is `0.966583`; it calls one actual versicolor flower virginica. The baseline accuracy is `0.333333` and macro-F1 is `0.166667`; it predicts setosa for every test flower. These are results for this particular split and feature pair, not evidence of field performance, causation, or discovery of an unknown species. See [`memo.md`](memo.md) for the calculations, per-species interpretation, limitations, and sources.

## Files

| Path | Purpose |
|---|---|
| [`knn_analysis.py`](knn_analysis.py) | One end-to-end script for loading data, splitting, validating, refitting, evaluating, printing results, and regenerating all five files below. |
| [`part2_arithmetic_check.py`](part2_arithmetic_check.py) | Optional NumPy-arithmetic check of the invented six-flower example in the memo; no scaler or KNN library is used for that example. |
| [`requirements.txt`](requirements.txt) | Python dependencies. |
| [`memo.md`](memo.md) | Hand calculations, explanations, measured results, large-data reasoning, and research process. |
| [`reports/knn_validation.csv`](reports/knn_validation.csv) | Seven rows with `k,train_accuracy,validation_accuracy`. |
| [`reports/test_summary.csv`](reports/test_summary.csv) | Two rows, `knn` and `baseline`, with accuracy and macro-F1. |
| [`reports/test_per_species.csv`](reports/test_per_species.csv) | Six model/species rows with precision, recall, F1, and integer support. |
| [`reports/k_selection.png`](reports/k_selection.png) | Training and validation accuracy versus `k`. |
| [`reports/test_confusion_matrices.png`](reports/test_confusion_matrices.png) | Labeled test confusion matrices for KNN and baseline; rows are actual species and columns are predictions. |

The CSV metrics are decimals between zero and one, not percentages. Each run overwrites these five generated files with results computed from the fixed split. The script resolves `reports/` relative to its own file, so it also writes to the correct place when invoked from another working directory.

## Before you begin

Install **Python 3.10 or newer** from [python.org/downloads](https://www.python.org/downloads/) and **Git** from [git-scm.com/downloads](https://git-scm.com/downloads). The macOS commands below use **Python 3.14**, the version tested for this repository. On Windows, select the Python installer's option to add Python to `PATH` if offered. Open **PowerShell** by searching for “PowerShell” in the Start menu. On macOS, open **Terminal** from Applications → Utilities (or Spotlight). Close and reopen the terminal after installing software.

Verify the installations in that terminal:

| Windows PowerShell | macOS Terminal |
|---|---|
| `python --version` | `python3.14 --version` (or `python3 --version` for another supported version) |
| `git --version` | `git --version` |

If you prefer an editor, install [Visual Studio Code](https://code.visualstudio.com/), select **File → Open Folder** after cloning, and open **Terminal → New Terminal**. After creating `.venv`, select its Python interpreter using **Command Palette → Python: Select Interpreter** (choose `.venv/bin/python` on macOS or `.venv\Scripts\python.exe` on Windows). VS Code is optional; the commands below work in PowerShell or Terminal by themselves. In each command block, work from the directory stated immediately above it. A command prompt displays the current directory; `pwd` prints it in either shell.

### Repository URL

The [student repository](https://github.com/Nuquynhchaunguyen/iris-species-classification-k-nearest-neighbors) is separate from the linked [course KNN materials](https://github.com/shylasolis/machine-learning-pathway/tree/main/Project_02_Classification/k-nearest-neighbors). The clone commands below name the local folder `iris-species-classification-knn`; GitHub's repository name is longer. Before independent validation, push the final files and check that the URL opens for your classmate.

Repository URL: `https://github.com/Nuquynhchaunguyen/iris-species-classification-k-nearest-neighbors`

## Clone and run on Windows PowerShell

1. Start PowerShell. These commands begin in your **Documents** folder and clone the repository into a new local folder named `iris-species-classification-knn`.

   ```powershell
   cd $HOME\Documents
   git clone https://github.com/Nuquynhchaunguyen/iris-species-classification-k-nearest-neighbors.git iris-species-classification-knn
   cd iris-species-classification-knn
   ```

2. You are now in the repository root, the folder containing `README.md` and `knn_analysis.py`. Create and activate an isolated Python environment, install dependencies, and run the script:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python knn_analysis.py
   ```

   To check the memo's invented-record arithmetic separately, run `python part2_arithmetic_check.py` from the same repository root.

   If PowerShell blocks activation, use the environment's interpreter directly **from the same repository root**. This does not require changing an execution policy:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe knn_analysis.py
   ```

3. The script prints the dataset and partition counts, all seven validation results, the chosen `k`, and readable test metric tables. To inspect the files, open the `reports` folder in File Explorer or VS Code. A PNG opens as an image; a CSV can be read in VS Code, Excel, or a text editor. PowerShell can print a CSV with `Get-Content reports\test_summary.csv`.

## Clone and run on macOS Terminal

1. Start Terminal. These commands begin in your **Documents** folder and clone the repository into a new local folder named `iris-species-classification-knn`.

   ```bash
   cd "$HOME/Documents"
   git clone https://github.com/Nuquynhchaunguyen/iris-species-classification-k-nearest-neighbors.git iris-species-classification-knn
   cd iris-species-classification-knn
   ```

2. You are now in the repository root. Create and activate an isolated Python environment, install dependencies, and run the script:

   ```bash
   python3.14 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python knn_analysis.py
   ```

   To check the memo's invented-record arithmetic separately, run `python part2_arithmetic_check.py` from the same repository root. If using another supported Python version, replace `python3.14` in the environment-creation command with that version's `python3` command.

3. Open `reports/` in Finder or VS Code. Double-click the PNGs to view the plots; a CSV opens in a spreadsheet or text editor. From the repository root, `cat reports/test_summary.csv` displays the summary in Terminal.

The environment is local to this folder and is excluded from Git. On a later visit, enter the repository root again and reactivate it, or run `.venv/bin/python knn_analysis.py` on macOS or `.\.venv\Scripts\python.exe knn_analysis.py` on Windows. If the environment is active, `deactivate` exits it.

## Expected run and reproducibility

The analysis was checked on macOS with **Python 3.14.10, NumPy 2.5.4, Matplotlib 3.11.2, and scikit-learn 1.9.1**. A warm run took about **1 second**; the first run took about **38 seconds** while Matplotlib built its font cache. Python 3.13.9 with NumPy 2.3.5, Matplotlib 3.10.6, and scikit-learn 1.7.2 produced the same reported metrics. Python 3.10+ with the packages in `requirements.txt` is the supported setup. Floating-point display in the last decimal can vary slightly across versions. Equal-distance neighbor selection at a `k` boundary can also depend on row order and library version; the split order is fixed and the script prints versions to investigate a discrepancy.

Successful execution creates or replaces **exactly these five generated outputs** in `reports/`:

```text
reports/knn_validation.csv
reports/test_summary.csv
reports/test_per_species.csv
reports/k_selection.png
reports/test_confusion_matrices.png
```

The script uses a fixed random state and never changes `k` in response to test results. Training and validation scores describe the training-only candidate models; test scores describe the separately refitted model. The baseline uses the same 120 development rows but no scaler. The test set is used only in the final evaluation.

## Troubleshooting

| Problem | What to check |
|---|---|
| `python` is not recognized on Windows | Reopen PowerShell after installing Python. Check `py -3 --version`; if it works, use `py -3 -m venv .venv` to create the environment. Otherwise reinstall Python from python.org with its `PATH` option, then reopen PowerShell. |
| `python3` or `git` is not recognized on macOS | Install from the official links above, reopen Terminal, and repeat the version checks. macOS may offer Command Line Tools when `git --version` is first run. |
| `ModuleNotFoundError` for `numpy`, `matplotlib`, or `sklearn` | Run the dependency-install command with the **same environment interpreter** used to run the script. In PowerShell, the direct `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` form works even if activation is blocked. |
| `requirements.txt` or `knn_analysis.py` cannot be found | Run `pwd`; it should end in `iris-species-classification-knn`. Use `cd` to enter that folder, then run the command again. |
| Clone reports “repository not found” | Confirm the repository URL above opens in a browser. The owner must push the final files and grant the reviewer access before validation. |
| Output numbers differ | Confirm the selected Python environment and package versions, that no script settings were changed, and that the script ran completely. Tiny decimal differences are possible; investigate larger differences before reporting successful replication. |


