import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from svc import KernelSVC
from utils import load_train_data


def cross_validate(
    data_path: str,
    labels_path: str,
    kernel_path: str,
    n_splits: int = 5,
    C_values: list = None,
    class_weights: bool = False,
    output_csv: str = None,
):
    if output_csv is None:
        output_csv = os.path.join(
            "results",
            os.path.basename(kernel_path).replace("_train.pkl", f"_{'' if class_weights else 'un'}balanced.csv"),
        )
    if not os.path.exists("results"):
        os.makedirs("results")

    # Loads list of graphs and training labels
    train_graphs, train_labels = load_train_data(
        data_path=data_path,
        labels_path=labels_path,
    )
    N = len(train_graphs)

    # Labels to {-1, 1}
    y = np.array(train_labels).reshape(-1)
    y = np.where(y == 0, -1, y)

    # Loading Kernel matrix
    with open(kernel_path, "rb") as file:
        K = pkl.load(file)

    # Cross-validation logs
    logs = {
        "fold": [],
        "C": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "auc": [],
    }

    # K-Fold
    for C in C_values:
        skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
        for i, (train_index, test_index) in enumerate(skf.split(np.zeros(N), y)):
            # Extract training and test kernel matrix
            K_train = K[train_index, :]
            K_train = K_train[:, train_index]
            y_train = y[train_index]

            K_test = K[test_index, :]
            K_test = K_test[:, train_index]
            y_test = y[test_index]

            # SVC Classifier
            svc = KernelSVC(C=C, kernel=None)
            svc.fit(X=None, y=y_train, K_train=K_train, method="cvxopt", class_weights=class_weights)
            out = svc.predict(X=None, K=K_test)
            y_pred = 2 * (out > 0) - 1

            # Updates logs
            logs["fold"].append(i)
            logs["C"].append(C)
            logs["accuracy"].append(accuracy_score(y_test, y_pred))
            logs["precision"].append(precision_score(y_test, y_pred))
            logs["recall"].append(recall_score(y_test, y_pred))
            logs["auc"].append(roc_auc_score(y_test, out))

    df = pd.DataFrame(logs)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launches cross-validation experiments")
    parser.add_argument("--data_path", type=str, default="data/training_data.pkl", help="Path to training data.")
    parser.add_argument("--labels_path", type=str, default="data/training_labels.pkl", help="Path to training labels.")
    parser.add_argument("--kernel_path", type=str, default="", help="Path to precomputed kernel matrix.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validatoin folds.")
    parser.add_argument("-l", "--c_values", nargs="+", help="List of values for parameter C.", required=True)
    parser.add_argument("--class_weights", dest="class_weights", action="store_true")
    parser.add_argument("--no_class_weights", dest="class_weights", action="store_false")
    parser.set_defaults(class_weights=True)
    parser.add_argument("--output_csv", type=str, default=None, help="Name of the output csv file.")
    args = parser.parse_args()

    cross_validate(
        data_path=args.data_path,
        labels_path=args.labels_path,
        kernel_path=args.kernel_path,
        n_splits=args.n_splits,
        C_values=[float(item) for item in args.c_values],
        class_weights=args.class_weights,
        output_csv=args.output_csv,
    )
