"""
Copyright to FOA Authors ICML 2024
"""

import torch
import torch.nn as nn
import torch.jit

from models.prompt_resnet import PromptResNet
import cma
import numpy as np
import time
import os
import math

class FOA_ResNet(nn.Module):
    """test-time Forward Only Adaptation
    FOA devises both input level and output level adaptation.
    It avoids modification to model weights and adapts in a backpropogation-free manner.
    """
    def __init__(self, args, model:PromptResNet, fitness_lambda=40):
        super().__init__()
        self.args = args
        self.fitness_lambda = fitness_lambda

        self.model = model
        self.popsize = 27
        self.es = self._init_cma() # initialization for CMA-ES

        self.best_padding = model.padding.weight.data # 对应prompts
        self.num_features = model.num_features
        self.best_loss = np.inf
        self.final_loss = np.inf  # 添加这一行，用于记录最终的损失值
        self.hist_stat = None # which is used for calculating the shift direction in Eqn. (8)
        
        
    def _init_cma(self):
        """CMA-ES initialization"""
        popsize = self.popsize if self.popsize > 1 else self.popsize + 1
        cma_opts = {
            'seed': 2020,
            'popsize': popsize,
            'maxiter': -1,
            'verbose': -1,
        }

        es = cma.CMAEvolutionStrategy(self.model.padding.weight.data.cpu().flatten(), 1, inopts=cma_opts)
        return es

    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean
            
    def _get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-1] - self.hist_stat

    def forward(self, x):
        """calculating shift direction, Eqn. (8)"""
        shift_vector = self._get_shift_vector()

        self.best_loss, self.best_outputs, batch_means = np.inf, None, []

        if self.popsize > 1:
            paddings, losses = self.es.ask() + [self.best_padding.flatten().cpu()], []#生成popsize个子代，但这里把最优解也加进去了
        else:
            paddings, losses = self.es.ask(), []

        for j, padding in enumerate(paddings):
            self.model.padding.weight.data = torch.nn.Parameter(torch.tensor(padding, dtype=torch.float).
                                                        reshape_as(self.model.padding.weight.data).cuda())
            self.model.padding.requires_grad_(False)

            outputs, loss, batch_mean = forward_and_get_loss(x, self.model, self.fitness_lambda, self.train_info, shift_vector, self.imagenet_mask)
            batch_means.append(batch_mean[-self.num_features:].unsqueeze(0))

            if self.best_loss > loss.item():
                self.best_padding = self.model.padding.weight.data
                self.best_loss = loss.item()
                self.best_outputs = outputs
                outputs = None

            losses.append(loss.item())
            del outputs

            print(f'Solution:[{j+1}/{len(paddings)}], Loss: {loss.item()}')

        """CMA-ES updates, Eqn. (6)"""
        self.es.tell(paddings, losses)

        self.model.padding.weight.data = self.best_padding

        """Update overall test statistics, Eqn. (9)"""
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_means)

        # 在这里添加最终损失值的计算和记录
        self.final_loss= self.best_loss

        return self.best_outputs
    
    def obtain_origin_stat(self, train_loader):
        print('===> begin calculating mean and variance')
        # 创建保存目录
        save_dir = os.path.join('dataset', 'train_stats')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f'train_info_promptresnet.pt')
        self.model.eval()

        # 尝试加载已保存的统计信息
        if os.path.exists(save_path):
            print('===> 从文件加载预计算的均值和方差')
            saved_data = torch.load(save_path)
            self.train_info = saved_data['train_info']
        else:
            print('===> 开始计算均值和方差')
            features, layer_stds, layer_means = [], [], []
            with torch.no_grad():
                for i, dl in enumerate(train_loader):
                    images = dl[0].cuda()
                    feature = self.model.forward_features(images)
                    # features.append([_ for _ in feature])
                    features.append(feature)
                    if i == 24: break

            for i in range(len(features[0])):
                layer_features = [feature[i] for feature in features]
                layer_features = torch.cat(layer_features, dim=0).cuda()

                assert len(layer_features.shape) == 2
                if len(layer_features.shape) == 4: dim = (0,2,3)
                else: dim = (0)

                layer_stds.append(layer_features.std(dim=dim))
                layer_means.append(layer_features.mean(dim=dim))

            layer_stds = torch.cat(layer_stds, dim=0)
            layer_means = torch.cat(layer_means, dim=0)
            self.train_info = (layer_stds, layer_means)
            # 保存计算结果
            print('===> 保存计算结果到文件')
            torch.save({
                'train_info': self.train_info,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }, save_path)
        
        print('===> calculating mean and variance end')

    def reset(self):
        self.es = self._init_cma()
        self.hist_stat = None
        

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='mean').cuda()

@torch.no_grad()
def forward_and_get_loss(images, model:PromptResNet, fitness_lambda, train_info, shift_vector, imagenet_mask):
    features = model.forward_features_with_prompts(images)
    discrepancy_loss = 0

    loss_std, loss_mean = 0, 0
    si, ei = 0, 0
    """discrepancy loss for Eqn. (5)"""
    for i in range(len(features)):
        layer_features = features[i]
        si = ei
        ei = si + features[i].shape[1]
        if len(layer_features.shape) == 4:
            dim = (0,2,3)
        else:
            dim = (0)

        batch_std, batch_mean = layer_features.std(dim=dim), layer_features.mean(dim=dim)
        loss_std += criterion_mse(batch_std, train_info[0][si:ei])
        loss_mean += criterion_mse(batch_mean, train_info[1][si:ei])
    
    loss_std = loss_std / len(features)
    loss_mean = loss_mean / len(features)
    discrepancy_loss += loss_std + loss_mean
    
    cls_features = features[-1] # the output feature of average pooling layer
    output = model.model.fc(cls_features)

    """entropy loss for Eqn. (5)"""
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]
    entropy_loss = softmax_entropy(output).mean()
    loss = fitness_lambda * discrepancy_loss + entropy_loss

    """activation shifting, Eqn. (7)"""
    if shift_vector is not None:
        output = model.model.fc(cls_features + 1. * shift_vector)
        if imagenet_mask is not None:
            output = output[:, imagenet_mask]

    return output, loss, batch_mean

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