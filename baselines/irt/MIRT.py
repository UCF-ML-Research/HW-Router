import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error


def irt2pl(theta, a, b, *, F=np):
    """

    Parameters
    ----------
    theta
    a
    b
    F

    Returns
    -------

    Examples
    --------
    >>> theta = [1, 0.5, 0.3]
    >>> a = [-3, 1, 3]
    >>> b = 0.5
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    0.109...
    >>> theta = [[1, 0.5, 0.3], [2, 1, 0]]
    >>> a = [[-3, 1, 3], [-3, 1, 3]]
    >>> b = [0.5, 0.5]
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    array([0.109..., 0.004...])
    """
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))


class MIRTNet(nn.Module):
    def __init__(self, llm_input_dim, item_input_dim, latent_dim, a_range, theta_range, irf_kwargs=None):
        super(MIRTNet, self).__init__()
        self.llm_input_dim = llm_input_dim
        self.item_input_dim = item_input_dim
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Linear(llm_input_dim, latent_dim, bias=False)
        self.a = nn.Linear(item_input_dim, latent_dim, bias=False)
        self.b = nn.Linear(item_input_dim, 1, bias=False)
        self.a_range = a_range
        self.theta_range = theta_range

    def forward(self, llm, item):
        theta = torch.squeeze(self.theta(llm), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        if self.theta_range is not None:
            theta = self.theta_range * torch.sigmoid(theta)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        b = torch.squeeze(self.b(item), dim=-1)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        
        # Get the prediction
        pred = self.irf(theta, a, b, **self.irf_kwargs)
        
        # Return theta, a, b, and the prediction
        return pred, theta, a, b

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)


class MIRT():
    def __init__(self, llm_input_dim, item_input_dim, latent_dim, a_range=None, theta_range=None):
        super(MIRT, self).__init__()
        self.irt_net = MIRTNet(llm_input_dim, item_input_dim, latent_dim, a_range, theta_range)

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        self.irt_net = self.irt_net.to(device)
        loss_function = nn.BCELoss()

        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                llm, item, response = batch_data
                llm: torch.Tensor = llm.to(device)
                item: torch.Tensor = item.to(device)
                pred, theta, a, b = self.irt_net(llm, item)
                response: torch.Tensor = response.to(device)
                loss = loss_function(pred, response)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                rmse, mae, auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] rmse: %.6f, mae: %.6f, auc: %.6f, accuracy: %.6f" % (e, rmse, mae, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        loss_function = nn.BCELoss()
        losses = []
        
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            llm, item, response = batch_data
            llm: torch.Tensor = llm.to(device)
            item: torch.Tensor = item.to(device)
            pred, theta, a, b = self.irt_net(llm, item)
            response: torch.Tensor = response.to(device)
            loss = loss_function(pred, response)
            losses.append(loss.mean().item())
            
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        print("[Valid Loss] %.6f" % (float(np.mean(losses))))
        self.irt_net.train()
        return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), roc_auc_score(np.array(y_true)>= 0.5, np.array(y_pred)>= 0.5), accuracy_score(np.array(y_true) >= 0.5, np.array(y_pred) >= 0.5)

    def generate(self, llm, item, device="cpu"):
        self.irt_net = self.irt_net.to(device)
        llm: torch.Tensor = llm.to(device)
        item: torch.Tensor = item.to(device)
        pred, theta, a, b = self.irt_net(llm, item) 
        return pred.tolist()
    
    def get_theta(self, llm, item, device="cpu"):
        self.irt_net = self.irt_net.to(device)
        llm: torch.Tensor = llm.to(device)
        item: torch.Tensor = item.to(device)
        pred, theta, a, b = self.irt_net(llm, item)  
        return theta.tolist()
    
    def get_e(self, llm, item, device="cpu"):
        self.irt_net = self.irt_net.to(device)
        llm: torch.Tensor = llm.to(device)
        item: torch.Tensor = item.to(device)
        pred, theta, a, b = self.irt_net(llm, item)  # Unpack the returned values
        return a.tolist()
    
    def get_difficulty(self, llm, item, device="cpu"):
        self.irt_net = self.irt_net.to(device)
        llm: torch.Tensor = llm.to(device)
        item: torch.Tensor = item.to(device)
        pred, theta, a, b = self.irt_net(llm, item)  
        return b.tolist()
    

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath, map_location="cuda"))
        logging.info("load parameters from %s" % filepath)
