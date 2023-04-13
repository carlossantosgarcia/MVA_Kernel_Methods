import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd
from kernels import RBF, WeisfeilerLehmanKernel
from sklearn.model_selection import StratifiedKFold
from utils import load_train_data


def generate_submission(method: str):
    # Loads training data
    train_graphs, train_labels = load_train_data(
        data_path="data/training_data.pkl",
        labels_path="data/training_labels.pkl",
    )

    # Loads test data
    with open("data/test_data.pkl", "rb") as file:
        test_graphs = pkl.load(file)

    # Labels to {-1, 1}
    N = len(train_graphs)
    y = np.array(train_labels).reshape(-1)
    y = np.where(y == 0, -1, y)

    assert method in ("rbf", "wl"), "Supported methods: 'rbf', 'wl'."
    if method == "wl":
        models_folder = os.path.join("models", "WL_unnorm_method=rbf_depth=4_sigma=5.0_C=1.0_balanced")
        kernel = WeisfeilerLehmanKernel(method="rbf", depth=4, sigma=5.0)
        K_test = kernel.kernel_matrix(X=test_graphs, Y=train_graphs, normalize=False)

    elif method == "rbf":
        models_folder = os.path.join("models", "rbf_unnorm_sigma=5_C=0.5_balanced")
        kernel = RBF(sigma=5.0)
        X_train = kernel.compute_features(list_graphs=train_graphs)
        X_test = kernel.compute_features(list_graphs=test_graphs)
        K_test = kernel.kernel_matrix(X=X_test, Y=X_train)

    predictions = {}
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(np.zeros(N), y)):
        # Loads trained SVM
        with open(os.path.join(models_folder, f"fold_{i}.pkl"), "rb") as f:
            svc = pkl.load(f)

        # Test kernel
        K_test_temp = K_test[:, train_index]

        # Prediction
        test_pred = svc.predict(X=None, K=K_test_temp)

        # Adds test predictions
        predictions[f"fold_{i}"] = list(test_pred.reshape(-1))

    dataframe = pd.DataFrame(predictions)
    dataframe["Predicted"] = dataframe.mean(axis=1)
    dataframe.index += 1
    if not os.path.exists("submissions"):
        os.makedirs("submissions")
    dataframe["Predicted"].to_csv(os.path.join("submissions", f"{method}.csv"), index_label="Id")
    print(f"Submission saved to {os.path.join('submissions', f'{method}.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates submission for the selected method.")
    parser.add_argument("--method", type=str, default="rbf", help="Kernel to use.")
    args = parser.parse_args()

    generate_submission(method=args.method)
