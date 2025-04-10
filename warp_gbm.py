import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import pandas as pd
import node_kernel
from tqdm import tqdm

class WarpGBM(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20,
        min_split_gain=0.0,
        verbosity=False,
    ):
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.forest = None
        self.bin_edges = None  # shape: [num_features, num_bins-1] if using quantile binning
        self.base_prediction = None
        self.unique_eras = None
        self.device = "cuda"
        self.root_gradient_histogram = None
        self.root_hessian_histogram = None
        self.gradients = None
        self.root_node_indices = None
        self.bin_indices = None
        self.Y_gpu = None
        self.num_features = None
        self.num_samples = None
        self.out_feature = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.out_bin = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain

    def fit(self, X, y, era_id):
        self.bin_indices, era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = self.preprocess_gpu_data(X, y, era_id)
        self.num_samples, self.num_features = X.shape
        self.gradients = torch.zeros_like(self.Y_gpu)
        self.root_node_indices = torch.arange(self.num_samples, device=self.device)
        self.base_prediction = self.Y_gpu.mean().item()
        self.gradients += self.base_prediction
        self.split_gains = torch.zeros((self.num_features, self.num_bins - 1), device=self.device)
        self.forest = self.grow_forest()
        return self

    def compute_quantile_bins(self, X, num_bins):
        quantiles = torch.linspace(0, 1, num_bins + 1)[1:-1]  # exclude 0% and 100%
        bin_edges = torch.quantile(X, quantiles, dim=0)       # shape: [B-1, F]
        return bin_edges.T  # shape: [F, B-1]
    
    def preprocess_gpu_data(self, X_np, Y_np, era_id_np):
        self.num_samples, self.num_features = X_np.shape
        Y_gpu = torch.from_numpy(Y_np).type(torch.float32).to(self.device)
        era_id_gpu = torch.from_numpy(era_id_np).type(torch.int32).to(self.device)
        is_integer_type = np.issubdtype(X_np.dtype, np.integer)
        if is_integer_type:
            max_vals = X_np.max(axis=0)
            if np.all(max_vals < self.num_bins):
                print("Detected pre-binned integer input — skipping quantile binning.")
                bin_indices = torch.from_numpy(X_np).to(self.device).contiguous().to(torch.int8)
    
                # We'll store None or an empty tensor in self.bin_edges
                # to indicate that we skip binning at predict-time
                bin_edges = torch.arange(1, self.num_bins, dtype=torch.float32).repeat(self.num_features, 1)
                bin_edges = bin_edges.to(self.device)
                unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
                return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu
            else:
                print("Integer input detected, but values exceed num_bins — falling back to quantile binning.")
    
        print("Performing quantile binning on CPU...")
        X_cpu = torch.from_numpy(X_np).type(torch.float32)  # CPU tensor
        bin_edges_cpu = self.compute_quantile_bins(X_cpu, self.num_bins).type(torch.float32).contiguous()
        bin_indices_cpu = torch.empty((self.num_samples, self.num_features), dtype=torch.int8)
        for f in range(self.num_features):
            bin_indices_cpu[:, f] = torch.bucketize(X_cpu[:, f], bin_edges_cpu[f], right=False).type(torch.int8)
        bin_indices = bin_indices_cpu.to(self.device).contiguous()
        bin_edges = bin_edges_cpu.to(self.device)
        unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
        return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu

    def compute_histograms(self, bin_indices_sub, gradients):
        grad_hist = torch.zeros((self.num_features, self.num_bins), device=self.device, dtype=torch.float32)
        hess_hist = torch.zeros((self.num_features, self.num_bins), device=self.device, dtype=torch.float32)
    
        node_kernel.compute_histogram(
            bin_indices_sub,
            gradients,
            grad_hist,
            hess_hist,
            self.num_bins
        )
        return grad_hist, hess_hist

    def find_best_split(self, gradient_histogram, hessian_histogram):
        node_kernel.compute_split(
            gradient_histogram.contiguous(),
            hessian_histogram.contiguous(),
            self.num_features,
            self.num_bins,
            0.0,  # L2 reg
            1.0,  # L1 reg
            1e-6, # hess cap
            self.out_feature,
            self.out_bin
        )
        
        f = int(self.out_feature[0])
        b = int(self.out_bin[0])
        return (f, b)
    
    def grow_tree(self, gradient_histogram, hessian_histogram, node_indices, depth):
        if depth == self.max_depth:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": node_indices.numel()}
    
        parent_size = node_indices.numel()
        best_feature, best_bin = self.find_best_split(gradient_histogram, hessian_histogram)
    
        if best_feature == -1:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": parent_size}
    
        split_mask = (self.bin_indices[node_indices, best_feature] <= best_bin)
        left_indices = node_indices[split_mask]
        right_indices = node_indices[~split_mask]

        left_size = left_indices.numel()
        right_size = right_indices.numel()

        if left_size == 0 or right_size == 0:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": parent_size}

        if left_size <= right_size:
            grad_hist_left, hess_hist_left = self.compute_histograms( self.bin_indices[left_indices], self.residual[left_indices] )
            grad_hist_right = gradient_histogram - grad_hist_left
            hess_hist_right = hessian_histogram - hess_hist_left
        else:
            grad_hist_right, hess_hist_right = self.compute_histograms( self.bin_indices[right_indices], self.residual[right_indices] )
            grad_hist_left = gradient_histogram - grad_hist_right
            hess_hist_left = hessian_histogram - hess_hist_right

        new_depth = depth + 1
        left_child = self.grow_tree(grad_hist_left, hess_hist_left, left_indices, new_depth)
        right_child = self.grow_tree(grad_hist_right, hess_hist_right, right_indices, new_depth)
    
        return { "feature": best_feature, "bin": best_bin, "left": left_child, "right": right_child }

    def grow_forest(self):
        forest = [{} for _ in range(self.n_estimators)]
        self.training_loss = []
    
        for i in range(self.n_estimators):
            self.residual = self.Y_gpu - self.gradients
    
            self.root_gradient_histogram, self.root_hessian_histogram = \
                self.compute_histograms(self.bin_indices, self.residual)
    
            tree = self.grow_tree(
                self.root_gradient_histogram,
                self.root_hessian_histogram,
                self.root_node_indices,
                depth=0
            )
            forest[i] = tree
            loss = ((self.Y_gpu - self.gradients) ** 2).mean().item()
            self.training_loss.append(loss)
            print(f"🌲 Tree {i+1}/{self.n_estimators} - MSE: {loss:.6f}")
    
        print("Finished training forest.")
        return forest

    def predict(self, X_np, era_id_np=None, chunk_size=50000):
        """
        Vectorized predict using a padded layer-by-layer approach.
        We assume `flatten_forest_to_tensors` has produced
        self.flat_forest with "features", "thresholds", "leaf_values",
        all shaped [n_trees, max_nodes].
        """

        # 1) Convert X_np -> bin_indices
        is_integer_type = np.issubdtype(X_np.dtype, np.integer)
        if is_integer_type:
            max_vals = X_np.max(axis=0)
            if np.all(max_vals < self.num_bins):
                bin_indices = X_np.astype(np.int8)
            else:
                raise ValueError("Pre-binned integers must be < num_bins")
        else:
            X_cpu = torch.from_numpy(X_np).type(torch.float32)
            bin_indices = torch.empty((X_np.shape[0], X_np.shape[1]), dtype=torch.int8)
            bin_edges_cpu = self.bin_edges.to('cpu')
            for f in range(self.num_features):
                bin_indices[:, f] = torch.bucketize(X_cpu[:, f], bin_edges_cpu[f], right=False).type(torch.int8)
            bin_indices = bin_indices.numpy()

        # 2) Ensure we have a padded representation
        if not hasattr(self, 'flat_forest'):
            self.flat_forest = self.flatten_forest_to_tensors(self.forest)

        features_t   = self.flat_forest["features"]      # [n_trees, max_nodes], int16
        thresholds_t = self.flat_forest["thresholds"]    # [n_trees, max_nodes], int16
        values_t     = self.flat_forest["leaf_values"]   # [n_trees, max_nodes], float32
        max_nodes    = self.flat_forest["max_nodes"]

        n_trees = features_t.size(0)
        N       = bin_indices.shape[0]
        out     = np.zeros(N, dtype=np.float32)

        # 3) Process rows in chunks
        for start in tqdm(range(0, N, chunk_size)):
            end = min(start + chunk_size, N)
            chunk_np  = bin_indices[start:end]  # shape [chunk_size, F]
            chunk_gpu = torch.from_numpy(chunk_np).to(self.device)  # [chunk_size, F], int8

            # Accumulate raw (unscaled) leaf sums
            chunk_preds = torch.zeros((end - start,), dtype=torch.float32, device=self.device)

            # node_idx[i] tracks the current node index in the padded tree for row i
            node_idx = torch.zeros((end - start,), dtype=torch.int32, device=self.device)

            # 'active' is a boolean mask over [0..(end-start-1)], indicating which rows haven't reached a leaf
            active = torch.ones((end - start,), dtype=torch.bool, device=self.device)

            for t in range(n_trees):
                # We might re-initialize 'active' for each tree
                # if each tree is meant to start from the root for all rows
                # Or we can reuse 'active' from a previous iteration if we want
                # to represent some global state. Typically each tree is independent,
                # so we reset node_idx & active each time:
                node_idx.fill_(0)
                active.fill_(True)

                tree_features = features_t[t]     # shape [max_nodes], int16
                tree_thresh   = thresholds_t[t]   # shape [max_nodes], int16
                tree_values   = values_t[t]       # shape [max_nodes], float32

                # Up to self.max_depth+1 layers
                for _level in range(self.max_depth + 1):
                    # gather active row indices
                    active_idx = active.nonzero(as_tuple=True)[0]
                    if active_idx.numel() == 0:
                        break  # all rows are done in this tree

                    # gather node indices for those rows
                    current_node_idx = node_idx[active_idx]
                    # gather features, thresholds, and values for these nodes
                    f    = tree_features[current_node_idx]    # shape [#active], int16
                    thr  = tree_thresh[current_node_idx]      # shape [#active], int16
                    vals = tree_values[current_node_idx]      # shape [#active], float32

                    mask_no_node = (f == -2)
                    mask_leaf    = (f == -1)

                    # Mark leaf => add leaf_value
                    if mask_leaf.any():
                        leaf_rows = active_idx[mask_leaf]
                        chunk_preds[leaf_rows] += vals[mask_leaf]
                        active[leaf_rows] = False

                    # Mark no-node => no action, just mark inactive
                    if mask_no_node.any():
                        no_node_rows = active_idx[mask_no_node]
                        active[no_node_rows] = False

                    # The rest => internal nodes => do bin compare
                    mask_internal = (~mask_leaf & ~mask_no_node)
                    if mask_internal.any():
                        internal_rows = active_idx[mask_internal]
                        # gather feature, threshold for these internal rows
                        act_f   = f[mask_internal].long()
                        act_thr = thr[mask_internal]

                        # gather bin_indices => shape [#internal]
                        binvals = chunk_gpu[internal_rows, act_f]
                        go_left = (binvals <= act_thr)
                        # left = 2*i+1, right=2*i+2
                        new_left_idx  = current_node_idx[mask_internal] * 2 + 1
                        new_right_idx = current_node_idx[mask_internal] * 2 + 2

                        node_idx[internal_rows[go_left]]   = new_left_idx[go_left]
                        node_idx[internal_rows[~go_left]]  = new_right_idx[~go_left]
                    # end internal
                # end layer loop
            # end for each tree

            # Now chunk_preds has sum of all leaf values for each row in [start:end]
            out[start:end] = (
                self.base_prediction + self.learning_rate * chunk_preds
            ).cpu().numpy()

        return out




    def flatten_forest_to_tensors(self, forest):
        """
        Convert 'forest' (list of dict-based trees) into a fixed-size
        array representation for each tree, up to max_depth. Each tree
        is stored in a 'perfect binary tree' layout:

          - node 0 is the root
          - node i has children (2*i + 1) and (2*i + 2), if in range
          - feature=-2 => no node / invalid
          - feature=-1 => leaf node
          - otherwise => internal node with that feature
        """
        n_trees = len(forest)
        max_nodes = 2 ** (self.max_depth + 1) - 1  # total array size per tree

        # Allocate padded arrays on CPU first (for easy indexing), then we'll move to GPU
        feat_arr = np.full((n_trees, max_nodes), -2, dtype=np.int16)
        thresh_arr = np.full((n_trees, max_nodes), -2, dtype=np.int16)
        value_arr = np.zeros((n_trees, max_nodes), dtype=np.float32)

        def fill_padded(tree, tree_idx, node_idx, depth):
            """
            Recursively fill feat_arr, thresh_arr, value_arr for this 'tree'
            into slot 'node_idx'. If 'depth == self.max_depth', no children.
            If there's no node, we leave feature=-2.
            """
            # If it's a leaf
            if "leaf_value" in tree:
                feat_arr[tree_idx, node_idx] = -1
                thresh_arr[tree_idx, node_idx] = -1
                value_arr[tree_idx, node_idx] = tree["leaf_value"]
                return

            # Otherwise it's an internal node
            feat = tree["feature"]
            bin_th = tree["bin"]

            feat_arr[tree_idx, node_idx] = feat
            thresh_arr[tree_idx, node_idx] = bin_th
            # value_arr stays 0 for internal nodes

            if depth < self.max_depth:
                left_idx  = 2*node_idx + 1
                right_idx = 2*node_idx + 2
                fill_padded(tree["left"],  tree_idx, left_idx,  depth+1)
                fill_padded(tree["right"], tree_idx, right_idx, depth+1)
            else:
                # At max depth => children are not stored => remain -2
                pass

        # Fill each tree
        for t, root in enumerate(forest):
            fill_padded(root, t, 0, 0)

        # Now convert to torch Tensors on the same device as self.device
        features_t = torch.from_numpy(feat_arr).to(self.device)
        thresholds_t = torch.from_numpy(thresh_arr).to(self.device)
        leaf_values_t = torch.from_numpy(value_arr).to(self.device)

        # Return them in a dict, similar to your original style
        return {
            "features": features_t,       # [n_trees, max_nodes]
            "thresholds": thresholds_t,   # [n_trees, max_nodes]
            "leaf_values": leaf_values_t, # [n_trees, max_nodes]
            "max_nodes": max_nodes        # so we know how many slots
        }



    def save_model(self, path):
        payload = {
            "forest": self.forest,
            "base_prediction": self.base_prediction,
            "bin_edges": self.bin_edges.cpu(),  # move to CPU to save space
            "num_bins": self.num_bins,
            "num_features": self.num_features,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "min_split_gain": self.min_split_gain,
            "device": self.device,
        }
        torch.save(payload, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.forest = checkpoint["forest"]
        self.flat_forest = checkpoint.get("flat_forest", self.flatten_forest_to_tensors(self.forest))  # recompute if missing
        self.base_prediction = checkpoint["base_prediction"]
        self.bin_edges = checkpoint["bin_edges"].to(self.device)
        self.num_bins = checkpoint["num_bins"]
        self.num_features = checkpoint["num_features"]
        self.learning_rate = checkpoint["learning_rate"]
        self.max_depth = checkpoint["max_depth"]
        self.n_estimators = checkpoint["n_estimators"]
        self.min_child_weight = checkpoint["min_child_weight"]
        self.min_split_gain = checkpoint["min_split_gain"]
        self.device = checkpoint["device"]




