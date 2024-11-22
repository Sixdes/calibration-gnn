
import torch
from torch import nn, optim
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
import scipy 
import numpy as np

import sklearn
from sklearn.utils.validation import check_is_fitted


def RBS(data, probs, val_logits, val_labels, logits, num_bins):
    def create_adjacency_matrix(graph):
        index_1 = [edge[0] for edge in graph.edges()] + [
            edge[1] for edge in graph.edges()
        ]
        index_2 = [edge[1] for edge in graph.edges()] + [
            edge[0] for edge in graph.edges()
        ]
        values = [1 for edge in index_1]
        node_count = max(max(index_1) + 1, max(index_2) + 1)
        A = scipy.sparse.coo_matrix(
            (values, (index_1, index_2)),
            shape=(node_count, node_count),
            dtype=np.float32,
        )
        return A

    graph = to_networkx(data)
    A = create_adjacency_matrix(graph).todense()

    # Calculate agg. probs
    AP = A * probs
    AP = torch.tensor(AP)
    num_neighbors = A.sum(1)
    num_neighbors = torch.tensor(num_neighbors)
    AP[torch.where(num_neighbors == 0)[0]] = 1
    num_neighbors[torch.where(num_neighbors == 0)[0]] = 1
    y_pred = torch.tensor(probs).max(1)[1]
    AP = AP / num_neighbors.expand(AP.shape[0], AP.shape[1])
    conf_AP = []
    for i in range(AP.shape[0]):
        conf_AP.append(AP[i, y_pred[i]])
    sm_prob = np.array(conf_AP)

    # Calculate val and test bins_mask_list
    sm_val = sm_prob[data.val_mask.detach().cpu().numpy()]
    sm_TS_model = bin_mask_eqdis(num_bins, sm_val)
    bins_mask_list = sm_TS_model.get_samples_mask_bins()

    # sm_test = sm_prob[data.test_mask.detach().cpu().numpy()]
    sm_test = sm_prob
    sm_TS_model = bin_mask_eqdis(num_bins, sm_test)
    bins_mask_list_test = sm_TS_model.get_samples_mask_bins()

    # Learn temperature
    T_list = []
    for i in range(num_bins):
        TS_model = TemperatureScaling_bins()
        # print("val_logits type:", type(val_logits))
        # print(type(bins_mask_list[i][0]))
        # val_logits = torch.tensor(val_logits)
        # val_labels = torch.tensor(val_labels)
        T = TS_model.fit(val_logits[bins_mask_list[i].detach().cpu().numpy()], val_labels[bins_mask_list[i].detach().cpu().numpy()])
        T_list.append(torch.tensor(T[1]))

    def get_rescaled_logits(T_list, logits, bins_mask_list):
        T = torch.zeros_like(logits)
        for i in range(num_bins):
            # The i-th bin logits
            logits_i = logits[bins_mask_list[i]]
            # Expand temperature to match the size of logits
            T_i = T_list[i].expand(logits_i.size(0), logits_i.size(1))
            T[bins_mask_list[i], :] = T_i.float()
        logits0 = logits / T
        return logits0
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cal_probs_test = torch.softmax(
        get_rescaled_logits(T_list, torch.tensor(logits), bins_mask_list_test), 1
    ).to(device)

    return cal_probs_test

class bin_mask_eqdis(nn.Module):
    def __init__(self, num_bins, sm_vector):
        super(bin_mask_eqdis, self).__init__()
        if not torch.is_tensor(sm_vector):
            self.sm_vector = torch.tensor(sm_vector)
        if torch.cuda.is_available():
            self.sm_vector = self.sm_vector.clone().cuda()
        self.num_bins = num_bins
        self.bins = []
        self.get_equal_bins()

    def get_equal_bins(self):
        for i in range(self.num_bins):
            self.bins.append(torch.tensor(1 / self.num_bins * (i + 1)))

    def get_samples_mask_bins(self):
        mask_list = []
        for i in range(self.num_bins):
            if i == 0:
                mask_list.append(self.sm_vector <= self.bins[i])
            else:
                mask_list.append(
                    (self.bins[i - 1] < self.sm_vector)
                    * (self.sm_vector <= self.bins[i])
                )
        return mask_list


class CalibrationMethod(sklearn.base.BaseEstimator):
    """
    A generic class for probability calibration
    A calibration method takes a set of posterior class probabilities and transform them into calibrated posterior
    probabilities. Calibrated in this sense means that the empirical frequency of a correct class prediction matches its
    predicted posterior probability.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.
        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def predict(self, X):
        """
        Predict the class of new samples after scaling. Predictions are identical to the ones from the uncalibrated
        classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.
        Returns
        -------
        C : array, shape (n_samples,)
            The predicted classes.
        """
        return np.argmax(self.predict_proba(X), axis=1)


class TemperatureScaling_bins(CalibrationMethod):
    def __init__(self, T_init=1, verbose=False):
        super().__init__()
        if T_init <= 0:
            raise ValueError("Temperature not greater than 0.")
        self.T_init = T_init
        self.verbose = verbose

    def fit(self, X, y):

        # Define objective function (NLL / cross entropy)
        def objective(T):
            # Calibrate with given T
            P = scipy.special.softmax(X / T, axis=1)

            # Compute negative log-likelihood
            P_y = P[np.array(np.arange(0, X.shape[0])), y]
            tiny = np.finfo(np.float32).tiny  # to avoid division by 0 warning
            NLL = -np.sum(np.log(P_y + tiny))
            return NLL

        # Derivative of the objective with respect to the temperature T
        def gradient(T):
            # Exponential terms
            E = np.exp(X / T)

            # Gradient
            dT_i = (
                np.sum(
                    E * (X - X[np.array(np.arange(0, X.shape[0])), y].reshape(-1, 1)),
                    axis=1,
                )
            ) / np.sum(E, axis=1)
            grad = -dT_i.sum() / T ** 2
            return grad

        # Optimize
        self.T = scipy.optimize.fmin_bfgs(
            f=objective, x0=self.T_init, fprime=gradient, gtol=1e-06, disp=self.verbose
        )[0]

        # Check for T > 0
        if self.T <= 0:
            raise ValueError("Temperature not greater than 0.")

        return self, self.T

    def predict_proba(self, X):

        # Check is fitted
        check_is_fitted(self, "T")
        # Transform with scaled softmax
        return scipy.special.softmax(X / self.T, axis=1)

def produce_logits(gnn, data, device):
    test_logits_list = []
    test_labels_list = []
    gnn.to(device)
    data = data.to(device)

    with torch.no_grad():
        gnn.eval()
        logits = gnn(data.x, data.edge_index)[data.test_mask]
        test_logits_list.append(logits)
        test_labels_list.append(data.y[data.test_mask])
        test_logits = torch.cat(test_logits_list).to(device)
        test_labels = torch.cat(test_labels_list).to(device)
    val_logits_list = []
    val_labels_list = []
    with torch.no_grad():
        gnn.eval()
        logits = gnn(data.x, data.edge_index)[data.val_mask]
        val_logits_list.append(logits)
        val_labels_list.append(data.y[data.val_mask])
        val_logits = torch.cat(val_logits_list).to(device)
        val_labels = torch.cat(val_labels_list).to(device)
    val_probs = F.softmax(val_logits, dim=1).detach().cpu().numpy()
    val_labels = val_labels.detach().cpu().numpy()
    test_probs = F.softmax(test_logits, dim=1).detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    val_logits, test_logits = (
        val_logits.detach().cpu().numpy(),
        test_logits.detach().cpu().numpy(),
    )
    with torch.no_grad():
        gnn.eval()
        logits = gnn(data.x, data.edge_index)
    logits_list = []
    logits_list.append(logits)
    logits = torch.cat(logits_list).to(device)
    probs = F.softmax(logits, 1).detach().cpu().numpy()

    return (
        val_probs,
        test_probs,
        val_logits,
        test_logits,
        val_labels,
        test_labels,
        logits.detach().cpu().numpy(),
        probs,
    )