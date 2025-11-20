import torch
import torch.nn as nn
import torch.quantization as quant
import copy

try:
    from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
    from mqbench.prepare_by_platform import BackendType           # contain various Backend, contains Academic.
    from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
    from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
    from mqbench.advanced_ptq import ptq_reconstruction
except:
    print("MQBench is not installed")

extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'MSEObserver',                              # custom weight observer
        'a_observer': 'EMAMSEObserver',                              # custom activation observer
        'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': 8,                                             # custom bitwidth for weight,
            'symmetry': False,                                    # custom whether quant is symmetric for weight,
            'per_channel': True,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': 8,                                             # custom bitwidth for activation,
            'symmetry': False,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
        }
    }
}

class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")


# the config of Adround quantization
ptq_reconstruction_config = AttrDict({
    'pattern': 'layer', # layer for adround, block for qdrop
    'scale_lr': 4.0e-5,
    'warm_up': 0.2,
    'weight': 0.01,
    'max_count': 20000,
    'b_range': [20, 2],
    'keep_gpu': True,
    'round_mode': 'learned_hard_sigmoid',
    'prob': 1.0,
})

def ptq_with_mqbench(fp_model, calib_loader, train_loader, bit=8, resume=False):
    """
    The model is Fake quantized and the floating convolution layer is preserved.
    :param model: model to be quantized
    :param calib_loader: calibration dataloader
    :param train_loader: train dataloader
    :return: the fake quantized model
    """
    extra_config['extra_qconfig_dict']['w_qscheme']['bit'] = bit
    extra_config['extra_qconfig_dict']['a_qscheme']['bit'] = bit
    if bit == 4:
        extra_config['extra_qconfig_dict']['w_fakequantize'] = 'AdaRoundFakeQuantize'
        extra_config['extra_qconfig_dict']['a_fakequantize'] = 'FixedFakeQuantize'
    elif bit == 3:
        extra_config['extra_qconfig_dict']['w_fakequantize'] = 'AdaRoundFakeQuantize'
        extra_config['extra_qconfig_dict']['a_fakequantize'] = 'FixedFakeQuantize'
    elif bit == 2:
        # use W2A4 format
        extra_config['extra_qconfig_dict']['a_qscheme']['bit'] = 4 
        extra_config['extra_qconfig_dict']['w_fakequantize'] = 'AdaRoundFakeQuantize'
        extra_config['extra_qconfig_dict']['a_fakequantize'] = 'FixedFakeQuantize'

    model = copy.deepcopy(fp_model)
    #! 1. trace model and add quant nodes for model on Academic Backend
    model = prepare_by_platform(model, BackendType.Academic, extra_config)

    #! 2. turn on calibration, ready for gathering data
    enable_calibration(model)

    # Calibrate model: Run a forward propagation using calibration data
    # the observer will automatically record the statistic information
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for inputs, _ in calib_loader:
            model(inputs.cuda())

    if not resume and bit <= 4:
        # ptq_reconstruction loop
        stacked_tensor = []
        # add ptq_reconstruction data to stack
        for inputs, _ in train_loader:
            stacked_tensor.append(inputs.cuda())
        
        # start ptq_reconstruction
        model = ptq_reconstruction(model, stacked_tensor, ptq_reconstruction_config)
    
    #! 3. turn on actually quantization, ready for simulating Backend inference
    # disable the observer of the quantized model
    enable_quantization(model)

    # don't convert the model to int8
    # directly returns the model containing the FakeQuant module 
    return model



def disable_observers(model):
    """
    disable the Observer to avoid updating the quantization parameter 
    """
    for name, module in model.named_modules():
        if isinstance(module, quant.FakeQuantize):
            # Remove the observer from the FakeQuantize
            # module.activation_post_process = None
            # disable the obersever to update the statistic
            # module.observer_enabled[0] = 0
            module.enable_fake_quant()
            module.disable_observer()

    return model