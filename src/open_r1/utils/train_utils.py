from torch import nn

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    if forbidden_layer_names is None:
        forbidden_layer_names = []
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(forbidden in f"{name}.{n}".lower() for forbidden in forbidden_layer_names)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k for k in model._parameters.keys() if not any(forbidden in k.lower() for forbidden in forbidden_layer_names)
    ]
    return result


def get_decay_parameter_names(model) -> list[str]:
    """
    Get all parameter names that weight decay will be applied to.

    This function filters out parameters in two ways:
    1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
    2. By parameter name patterns (containing 'bias', 'layernorm', or 'rmsnorm')
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS, ["bias", "layernorm", "rmsnorm"])
    return decay_parameters

