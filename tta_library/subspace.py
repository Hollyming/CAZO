"""
Copyright to Tent Authors ICLR 2021 Spotlight
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.jit
import numpy as np
import time
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Subspace(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer,
                 n_components=10, batch_step=50,w_num=1500, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.batch_num = 0
        self.moving_W = []
        self.w_num = w_num
        self.n_components = n_components
        self.batch_step = batch_step
        self.temp = None
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        

    def forward(self, x, y=None):
        outputs = self.forward_and_adapt(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, ):
        if self.temp is None:
            self.temp = get_model_param_vec(self.model)

        outputs = self.model(x)
        # adapt
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()

        model_state, optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        if self.batch_num % self.batch_step == 0 and not self.batch_num == 0:  # 每self.batch_step个
            self.moving_W.append(get_model_grad_vec1(self.model))
            # 每在W中存储一次权重，就删除最前面的一次权重
            if len(self.moving_W) > self.w_num:
                self.moving_W.pop(0)
        elif self.batch_num == 0:  # 第一个batch存一次权重
            self.moving_W.append(self.temp)

        # if False:
        if len(self.moving_W) >= self.n_components:
            W = np.array(self.moving_W)
            W = W - np.mean(W, axis=0)  # 减去均值
            # PCA
            starttime = time.time()  # 记一下计算pca要用的时间
            pca = PCA(n_components=self.n_components)
            pca.fit_transform(W)  # note: 这一步对w进行了中心化和降维
            endtime = time.time()
            P = np.array(pca.components_)
            P = torch.from_numpy(P).to(device)
            # torch.save(P, corruption_type + '_' + 'subspace_matrix.pt')
            load_model_and_optimizer(self.model, self.optimizer,
                                     model_state, optimizer_state)
            gk = get_model_grad_vec(self.model)
            P_SGD(self.model, self.optimizer, gk, P)  # 子空间投影
        else:
            self.optimizer.step()
        self.batch_num += 1
        self.optimizer.zero_grad()
        return outputs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.jit.script
def energy(x: torch.Tensor) -> torch.Tensor:
    """Energy calculation from logits."""
    temprature = 1
    x = -(temprature*torch.logsumexp(x / temprature, dim=1))
    if torch.rand(1) > 0.95:
        print(x.mean(0).item())
    return x

def copy_model_only(model):
    source_model = deepcopy(model)
    for param in source_model.parameters():
        param.detach_()
    return source_model

@torch.enable_grad()  # ensure grads in possible no grad contself.model for testing
def forward_and_adapt(x, model, optimizer, imagenet_mask):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    if imagenet_mask is not None:
        outputs = outputs[:, imagenet_mask]
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

def P_SGD(model, optimizer, grad, P, retrun_grad=False):
  gk = torch.mm(P, grad.reshape(-1, 1))
  grad_proj = torch.mm(P.transpose(0, 1), gk)
  update_grad(model, grad_proj)
  gk_proj = get_model_grad_vec(model)  # 返回投影后的梯度
  optimizer.step()
  return gk_proj if retrun_grad else None

def update_grad(model, grad_vec):
  idx = 0
  for name, param in model.named_parameters():
    # 如果梯度为空，则加入全0的梯度
    if param.requires_grad:
        if param.grad is None:
          param.grad = torch.zeros_like(param)
        arr_shape = param.grad.shape
        size = 1
        for i in range(len(list(arr_shape))):
          size *= arr_shape[i]
        param.grad.data = grad_vec[idx:idx + size].reshape(arr_shape)
        idx += size
          
def get_model_param_vec(model):
  vec = []
  for name, param in model.named_parameters():
      if param.requires_grad:
        vec.append(param.detach().cpu().numpy().reshape(-1))
  return np.concatenate(vec, 0)

def get_model_grad_vec1(model):
  vec = []
  for name, param in model.named_parameters():
    # 如果梯度为空，则加入全0的梯度
    if param.requires_grad:
        vec.append(param.grad.detach().cpu().numpy().reshape(-1))
  return np.concatenate(vec, 0)


def get_model_grad_vec(model):
  # Return the model grad as a vector

  vec = []
  for name, param in model.named_parameters():
    # 如果梯度为空，则加入全0的梯度
    if param.requires_grad:
        vec.append(param.grad.detach().reshape(-1))
    # if param.grad is None:
    #   vec.append(torch.zeros_like(param).reshape(-1))
    # else:
    #   vec.append(param.grad.detach().reshape(-1))
  return torch.cat(vec, 0)