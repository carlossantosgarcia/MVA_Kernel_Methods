import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from svc import KernelSVC
from utils import load_train_data


def generate_submission(
    data_path: str,
    labels_path: str,
    K_train_path: np.ndarray,
    K_test_path: np.ndarray,
    C: float,
    class_weights: bool,
    n_splits: int,
    output_name: str,
    save_models: bool,
):
    """Generates CSV submission file.

    Args:
        data_path (str): Path to the list of training graphs.
        labels_path (str): Path to the list of training labels.
        K_train_path (np.ndarray): Path to the precomputed train kernel matrix.
        K_test_path (np.ndarray): Path to the precomputed test kernel matrix.
        C (float): SVM regularization parameter.
        class_weights (bool): If true, weights each class inverse proportional to its size.
        n_splits (int): Number of splits used for cross-validation.
        output_name (str): Name of the output file.
        save_models (bool): If true, saves a model per fold for quicker predictions.
    """
    if output_name is None:
        output_name = os.path.basename(K_train_path).replace(
            "_train.pkl", f"_C={C}_{'' if class_weights else 'un'}balanced.csv"
        )
    if not os.path.exists("submissions"):
        os.makedirs("submissions")
    if save_models:
        output_folder = os.path.join("models", output_name.replace(".csv", ""))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

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
    with open(K_train_path, "rb") as file:
        K = pkl.load(file)
        print("K shape:", K.shape)
    with open(K_test_path, "rb") as file:
        K_test = pkl.load(file)
        print("K_test shape:", K_test.shape)

    logs = {
        "fold": [],
        "C": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "auc": [],
    }
    predictions = {}
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(np.zeros(N), y)):
        # Extract training and test kernel matrix
        K_train = K[train_index, :]
        K_train = K_train[:, train_index]
        y_train = y[train_index]

        # Validation data
        K_val = K[val_index, :]
        K_val = K_val[:, train_index]
        y_val = y[val_index]

        # Test kernel
        K_test_temp = K_test[:, train_index]

        # SVC Classifier
        svc = KernelSVC(C=C, kernel=None)
        svc.fit(X=None, y=y_train, K_train=K_train, method="cvxopt", class_weights=class_weights)
        if save_models:
            svc.K_train = None
            with open(os.path.join(output_folder, f"fold_{i}.pkl"), "wb") as f:
                pkl.dump(svc, f)
        test_pred = svc.predict(X=None, K=K_test_temp)
        val_pred = svc.predict(X=None, K=K_val)
        y_val_pred = 2 * (val_pred > 0) - 1

        # Updates logs
        logs["fold"].append(i)
        logs["C"].append(C)
        logs["accuracy"].append(accuracy_score(y_val, y_val_pred))
        logs["precision"].append(precision_score(y_val, y_val_pred))
        logs["recall"].append(recall_score(y_val, y_val_pred))
        logs["auc"].append(roc_auc_score(y_val, val_pred))

        # Adds test predictions
        predictions[f"fold_{i}"] = list(test_pred.reshape(-1))

    dataframe = pd.DataFrame(predictions)
    dataframe["Predicted"] = dataframe.mean(axis=1)
    dataframe.index += 1
    if not output_name.endswith(".csv"):
        output_name += ".csv"
    dataframe.to_csv(os.path.join("submissions", f"complete_{output_name}"), index_label="Id")
    dataframe["Predicted"].to_csv(os.path.join("submissions", f"ensemble_{output_name}"), index_label="Id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launches the creation of a submission file")
    parser.add_argument("--data_path", type=str, default="data/training_data.pkl", help="Path to training data.")
    parser.add_argument("--labels_path", type=str, default="data/training_labels.pkl", help="Path to training labels.")
    parser.add_argument("--K_train_path", type=str, default="", help="Path to precomputed train kernel matrix.")
    parser.add_argument("--K_test_path", type=str, default="", help="Path to precomputed test kernel matrix.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validatoin folds.")
    parser.add_argument("--C", type=float, default=100, help="SVM regularization parameter.")
    parser.add_argument("--class_weights", dest="class_weights", action="store_true")
    parser.add_argument("--no_class_weights", dest="class_weights", action="store_false")
    parser.set_defaults(class_weights=True)
    parser.add_argument("--output_name", type=str, default=None, help="Name of the output csv file.")
    parser.add_argument("--save_models", dest="save_models", action="store_true")
    parser.add_argument("--no_save_models", dest="save_models", action="store_false")
    parser.set_defaults(save_models=False)
    args = parser.parse_args()

    generate_submission(
        data_path=args.data_path,
        labels_path=args.labels_path,
        K_train_path=args.K_train_path,
        K_test_path=args.K_test_path,
        C=args.C,
        class_weights=args.class_weights,
        n_splits=args.n_splits,
        output_name=args.output_name,
        save_models=args.save_models,
    )
