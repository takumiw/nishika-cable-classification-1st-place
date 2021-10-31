"""Create image models"""
from logging import getLogger
from typing import Dict

import timm
from efficientnet_pytorch import EfficientNet as EfficientNetPytorch
from torch import nn

logger = getLogger(__name__)


def efficientnetv2_params(model_name: str = "efficientnetv2_rw_s") -> Dict[str, float]:
    """
    Map EfficientNet-V2 model name to parameter coefficients.
    """
    params_dict = {
        "efficientnetv2_rw_s": {"resolution": 384, "dropout": 0.2},
        "efficientnetv2_rw_m": {"resolution": 416, "dropout": 0.3},
    }
    return params_dict[model_name]


def efficientnet_params(model_name: str = "efficientnet_b0") -> Dict[str, float]:
    """
    Map EfficientNet model name to parameter coefficients.
    in_features are: 1024, 1280, 1408, 1536, 1792 for efficientnet_b0 to efficientnet_b4
    """
    params_dict = {
        # Coefficients: res,dropout
        "efficientnet_b0": {"resolution": 224, "dropout": 0.2},
        "efficientnet_b1": {"resolution": 240, "dropout": 0.2},
        "efficientnet_b2": {"resolution": 260, "dropout": 0.3},
        "efficientnet_b3": {"resolution": 300, "dropout": 0.3},
        "efficientnet_b4": {"resolution": 380, "dropout": 0.4},
        "efficientnet_b5": {"resolution": 456, "dropout": 0.4},
        "efficientnet_b6": {"resolution": 528, "dropout": 0.5},
        "efficientnet_b7": {"resolution": 600, "dropout": 0.5},
        "efficientnet_b8": {"resolution": 672, "dropout": 0.5},
        "efficientnet_l2": {"resolution": 800, "dropout": 0.5},
    }
    return params_dict[model_name]


def resnet_params(model_name: str = "resnet34d") -> Dict[str, float]:
    """
    Map ResNet model name to parameter coefficients.
    """
    params_dict = {
        # Coefficients: res,dropout
        "resnet18d": {"resolution": 224, "dropout": 0.2},
        "resnet34d": {"resolution": 224, "dropout": 0.2},
        "resnet50d": {"resolution": 224, "dropout": 0.3},
        "resnet101d": {"resolution": 256, "dropout": 0.3},
        "resnet152d": {"resolution": 256, "dropout": 0.4},
    }
    return params_dict[model_name]


def create_model(
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    hidden_dim: int = 512,
    do_dropout: bool = True,
    out_channels: int = 15,
):
    """
    Args:
        model_name (str): Selected from efficientnet_b[0-4], resnet18d
    """
    if "efficientnetv2" in model_name:
        model = timm.create_model(model_name=model_name, pretrained=pretrained)
        in_features = model.num_features
        dropout = efficientnetv2_params(model_name)["dropout"]
        model.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=out_channels),
        )
        logger.debug(f"Create {model_name=} where {pretrained=}, {hidden_dim=}, {out_channels=}")

    elif "efficientnet" in model_name:
        if model_name == "efficientnet_b5":
            model = EfficientNetPytorch.from_pretrained("efficientnet-b5")
            in_features = model._fc.in_features
            dropout = efficientnet_params(model_name)["dropout"]
            model._fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features=in_features, out_features=out_channels),
            )
            logger.info(
                f"Create {model_name=} from efficientnet_pytorch where {pretrained=}, {hidden_dim=}, {out_channels=}"
            )

        elif model_name == "efficientnet_b6":
            model = EfficientNetPytorch.from_pretrained("efficientnet-b6")
            in_features = model._fc.in_features
            dropout = efficientnet_params(model_name)["dropout"]
            model._fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features=in_features, out_features=out_channels),
            )
            logger.info(
                f"Create {model_name=} from efficientnet_pytorch where {pretrained=}, {hidden_dim=}, {out_channels=}"
            )

        elif model_name == "efficientnet_b7":
            model = EfficientNetPytorch.from_pretrained("efficientnet-b7")
            in_features = model._fc.in_features
            dropout = efficientnet_params(model_name)["dropout"]
            model._fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features=in_features, out_features=out_channels),
            )
            logger.info(
                f"Create {model_name=} from efficientnet_pytorch where {pretrained=}, {hidden_dim=}, {out_channels=}"
            )

        else:
            model = timm.create_model(model_name=model_name, pretrained=pretrained)
            in_features = model.num_features
            dropout = efficientnet_params(model_name)["dropout"]
            if hidden_dim:
                model.classifier = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_features=hidden_dim, out_features=out_channels),
                )
            else:
                if do_dropout:
                    model.classifier = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(in_features=in_features, out_features=out_channels),
                    )
                else:
                    model.classifier = nn.Sequential(
                        nn.Linear(in_features=in_features, out_features=out_channels),
                    )
            logger.debug(f"Create {model_name=} where {pretrained=}, {hidden_dim=}, {out_channels=}")

    elif "resnet" in model_name:
        model = timm.create_model(model_name=model_name, pretrained=pretrained)
        in_features = model.num_features
        dropout = resnet_params(model_name)["dropout"]
        if hidden_dim:
            model.fc = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(in_features=hidden_dim, out_features=out_channels, bias=True),
            )
        else:
            if do_dropout:
                model.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features=in_features, out_features=out_channels),
                )
            else:
                model.fc = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=out_channels),
                )
        logger.debug(f"Create {model_name=} where {pretrained=}, {hidden_dim=}, {out_channels=}")

    elif "vit" in model_name:
        model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=out_channels)
        logger.debug(f"Create {model_name=} where {pretrained=}, {out_channels=}")

    elif "gmlp" in model_name:
        model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=out_channels)
        logger.debug(f"Create {model_name=} where {pretrained=}, {out_channels=}")

    elif "nfnet" in model_name:
        model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=out_channels)
        logger.debug(f"Create {model_name=} where {pretrained=}, {out_channels=}")

    else:
        raise NameError(f"model_name {model_name} is not defined")

    return model
