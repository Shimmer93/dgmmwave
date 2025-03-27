import yaml
from argparse import Namespace
import sys
from collections import OrderedDict

def load_cfg(cfg):
    hyp = None
    if isinstance(cfg, str):
        with open(cfg, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    return Namespace(**hyp)


def merge_args_cfg(args, cfg):
    dict0 = vars(args)
    dict1 = vars(cfg)
    dict = {**dict0, **dict1}

    return Namespace(**dict)

def torch2numpy(tensor):
    return tensor.detach().cpu().numpy()

def import_with_str(module, name):
    return getattr(sys.modules[module], name)

def delete_prefix_from_state_dict(state_dict, prefix):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict