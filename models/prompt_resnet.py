import torch
import torch.nn as nn

def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3*dilation, groups=groups, bias=False, dilation=dilation)


class PromptResNet(nn.Module):
    def __init__(self,
                args,
                resnet):
        super().__init__()
        self.args = args
        self.model = resnet
        self.num_features = resnet.fc.in_features
        self.padding = conv7x7(3,3)

        for m in self.padding.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    
    def padX(self, x):
        return x + self.padding(x)
    
    def embed(self, x):
        if self.args.quant:
            assert hasattr(self.model, 'x_post_act_fake_quantizer')
            x = self.model.x_post_act_fake_quantizer(x)
        
        # insert prompt
        x = self.padX(x)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        if self.args.quant:
            assert hasattr(self.model, 'maxpool_post_act_fake_quantizer')
            x = self.model.maxpool_post_act_fake_quantizer(x)
        return x
        
    def forward(self, x):
        x = self.embed(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        x = self.model.layer4(x)
        x = self.model.forward_head(x)
        return x
    
    def forward_features(self, x):
        x = self.embed(x)
        if self.args.quant:
            return self.get_stages_last_quant_features(x)
        return self.get_stages_last_features(x)
    
    def get_stages_last_features(self, x):
        features = []
        x = self.model.layer1(x)
        # features.append(x)
        features.append(self.model.avgpool(x).reshape(x.size(0), -1))

        x = self.model.layer2(x)
        # features.append(x)
        features.append(self.model.avgpool(x).reshape(x.size(0), -1))

        x = self.model.layer3(x)
        # features.append(x)
        features.append(self.model.avgpool(x).reshape(x.size(0), -1))

        x = self.model.layer4(x)
        # features.append(x)

        x = self.model.avgpool(x)
        x = x.reshape(x.size(0), -1)
        features.append(x)

        return features

    def get_stages_last_quant_features(self, x):
        features = []
        # forward with quantization
        # forward for each layer
        layer_list = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        for layer_idx, layer in enumerate(layer_list):
            # forward for each block
            for idx in range(len(list(layer.children()))):
                identity = x
                out = layer.get_submodule(str(idx)).conv1(x)
                out = layer.get_submodule(str(idx)).bn1(out)
                out = layer.get_submodule(str(idx)).relu(out)
                # post_act_fake_quantizer
                if hasattr(self.model, f'layer{layer_idx+1}_{idx}_relu_post_act_fake_quantizer'):
                    out = self.model.get_submodule(f'layer{layer_idx+1}_{idx}_relu_post_act_fake_quantizer')(out)

                out = layer.get_submodule(str(idx)).conv2(out)
                out = layer.get_submodule(str(idx)).bn2(out)
                out = layer.get_submodule(str(idx)).relu(out)
                # post_act_fake_quantizer
                if hasattr(self.model, f'layer{layer_idx+1}_{idx}_relu_1_post_act_fake_quantizer'):
                    out = self.model.get_submodule(f'layer{layer_idx+1}_{idx}_relu_1_post_act_fake_quantizer')(out)

                out = layer.get_submodule(str(idx)).conv3(out)
                out = layer.get_submodule(str(idx)).bn3(out)

                if hasattr(layer.get_submodule(str(idx)), 'downsample'):
                    identity = layer.get_submodule(str(idx)).downsample.get_submodule('0')(x)
                    identity = layer.get_submodule(str(idx)).downsample.get_submodule('1')(identity)

                out += identity
                x = layer.get_submodule(str(idx)).relu(out)

                # # post_act_fake_quantizer
                if hasattr(self.model, f'layer{layer_idx+1}_{idx}_relu_2_post_act_fake_quantizer'):
                    x = self.model.get_submodule(f'layer{layer_idx+1}_{idx}_relu_2_post_act_fake_quantizer')(x)

            # record the last feature of each layer
            if layer_idx != len(layer_list) - 1:
                features.append(self.model.avgpool(x).reshape(x.size(0), -1))

        x = self.model.avgpool(x)
        x = x.reshape(x.size(0), -1)
        if hasattr(self.model, 'reshape_post_act_fake_quantizer'):
            x = self.model.reshape_post_act_fake_quantizer(x)
        
        # record the feature after reshape op
        features.append(x)

        return features
    
    def forward_features_with_prompts(self, x):
        x = self.embed(x)
        if self.args.quant:
            return self.get_stages_last_quant_features(x)
        return self.get_stages_last_features(x)