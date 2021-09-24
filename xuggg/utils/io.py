import torch
import torch.nn as nn


def get_state_dict_on_cpu(obj):
    '''
    args: model
    Move model.data to cpu
    state_dict.keys like ['Pconv1.weight', 'Pconv2.weight'...]
    state_dict[key] is weight.data
    '''
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)

    return state_dict


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    '''
    Save model and optimizers parameters
    args: path, ('generator', self.G), ('optimizer_G', self.optim_G), n_iter
    prefix like 'generator'
    model like CoarseNetEncoder((Pconv1): PartialConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)...)
    '''
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()

    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name)
    # print("Load: ", len(ckpt_dict))
    for prefix, model in models:
        # print("prefix: ", prefix)
        # print("model: ", model)
        assert isinstance(model, nn.Module)
        # print("ckpt_dict[prefix]: ", ckpt_dict[prefix])
        model.load_state_dict(ckpt_dict[prefix], strict=False)

    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])

    return ckpt_dict['n_iter']
