import argparse
import os
import pickle as pkl

import numpy as np
from kernels import (
    RBF,
    EfficientLabeledShortestPathKernel,
    NthOrderWalkKernel,
    UnlabeledShortestPathKernel,
    WeisfeilerLehmanKernel,
)
from utils import load_train_data


def compute_kernel_matrix(
    kernel: str,
    train_graphs: list,
    test_graphs: list,
    compute_test: bool = False,
    normalize: bool = True,
    length_walk: int = 3,
    sigma: float = 5.0,
    wl_method: str = "linear",
    wl_depth: int = 3,
):
    """Computes the K_train and K_test kernel matrices using the selected kernel.

    Args:
        kernel (str): Selected kernel.
        train_graphs (list): List of train graphs.
        test_graphs (list): List of test graphs.
        compute_test (bool, optional): If true computes K_test. Defaults to False.
        normalize (bool, optional): If true normalizes the kernel matrix. Defaults to True.
    """
    assert kernel in (
        "unlabeled_shortest_path",
        "labeled_shortest_path",
        "rbf",
        "nwalk",
        "wl",
    ), "Kernel not implemented !"
    assert wl_method in ("linear", "rbf"), "Possible methods are linear or rbf kernel with WL features."

    if kernel == "unlabeled_shortest_path":
        kname = "USP" + f"_{'' if normalize else 'un'}norm"
        kernel = UnlabeledShortestPathKernel()
        K_train = kernel.gram_matrix(train_graphs, normalize=normalize)
        if compute_test:
            K_test = kernel.kernel_matrix(X=test_graphs, Y=train_graphs, normalize=normalize)

    elif kernel == "labeled_shortest_path":
        kname = "LSP" + f"_{'' if normalize else 'un'}norm"
        kernel = EfficientLabeledShortestPathKernel()
        K_train = kernel.gram_matrix(train_graphs, normalize=normalize)
        if compute_test:
            K_test = kernel.kernel_matrix(X=test_graphs, Y=train_graphs, normalize=normalize)

    elif kernel == "rbf":
        assert compute_test, "Test matrix using RBF needs to be normalized using training parameters !"
        kname = "rbf" + f"_{'' if normalize else 'un'}norm_sigma={sigma}"
        kernel = RBF(sigma=sigma)

        X_train = kernel.compute_features(list_graphs=train_graphs)
        if normalize:
            val_max = np.max(X_train, axis=0)
            val_max = np.where(val_max == 0, 1, val_max)
            X_train = X_train / val_max
        K_train = kernel.kernel_matrix(X=X_train, Y=X_train)
        if compute_test:
            X_test = kernel.compute_features(list_graphs=test_graphs)
            if normalize:
                X_test = X_test / val_max
            K_test = kernel.kernel_matrix(X=X_test, Y=X_train)

    elif kernel == "nwalk":
        kname = "nwalk" + f"_{'' if normalize else 'un'}norm_n={length_walk}"
        kernel = NthOrderWalkKernel(n=length_walk)
        K_train = kernel.gram_matrix(train_graphs, normalize=normalize)
        if compute_test:
            K_test = kernel.kernel_matrix(X=test_graphs, Y=train_graphs, normalize=normalize)

    elif kernel == "wl":
        kname = "WL" + f"_{'' if normalize else 'un'}norm_method={wl_method}_depth={wl_depth}_sigma={sigma}"
        kernel = WeisfeilerLehmanKernel(method=wl_method, depth=wl_depth, sigma=sigma)
        K_train = kernel.gram_matrix(train_graphs, normalize=normalize)
        if compute_test:
            K_test = kernel.kernel_matrix(X=test_graphs, Y=train_graphs, normalize=normalize)

    if not os.path.exists("precomputed_kernels"):
        os.makedirs("precomputed_kernels")

    with open(os.path.join("precomputed_kernels", kname + "_train.pkl"), "wb") as f:
        K_train.dump(f)
    with open(os.path.join("precomputed_kernels", kname + "_test.pkl"), "wb") as f:
        K_test.dump(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launches kernel computations")
    parser.add_argument("--kernel", type=str, default="rbf", help="Graph kernel to use.")
    parser.add_argument("--data_path", type=str, default="data/training_data.pkl", help="Path to training data.")
    parser.add_argument("--labels_path", type=str, default="data/training_labels.pkl", help="Path to training labels.")
    parser.add_argument("--test_path", type=str, default="data/test_data.pkl", help="Path to test data.")
    parser.add_argument("--compute_test", dest="compute_test", action="store_true")
    parser.add_argument("--no_compute_test", dest="compute_test", action="store_false")
    parser.set_defaults(compute_test=False)
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument("--length_walk", type=int, default=3, help="Order for walk kernel.")
    parser.add_argument("--sigma", type=float, default=5.0, help="Sigma value for RBF kernel.")
    parser.add_argument("--wl_method", type=str, default="linear", help="Method for WL computation.")
    parser.add_argument("--wl_depth", type=int, default=3, help="Iterations of WL procedure.")

    args = parser.parse_args()

    # Loads training data
    train_graphs, _ = load_train_data(
        data_path=args.data_path,
        labels_path=args.labels_path,
    )

    # Loads test data
    with open(args.test_path, "rb") as file:
        test_graphs = pkl.load(file)

    compute_kernel_matrix(
        kernel=args.kernel,
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        compute_test=args.compute_test,
        normalize=args.normalize,
        length_walk=args.length_walk,
        sigma=args.sigma,
        wl_method=args.wl_method,
        wl_depth=args.wl_depth,
    )
