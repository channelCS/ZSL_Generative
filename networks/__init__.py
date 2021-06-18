import importlib
from utils.logger import get_root_logger


def create_model(opt):
    """Create model.
    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt["model_type"]
    module = importlib.import_module(f"networks.{model_type}_model")

    # dynamic instantiation
    model_cls = getattr(module, model_type, None)
    if model_cls is None:
        raise ValueError(f"Model {model_type} is not found.")

    model = model_cls(opt)

    logger = get_root_logger()
    logger.info(f"Model [{model.__class__.__name__}] is created.")
    return model
