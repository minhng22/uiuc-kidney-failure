from pkgs.models.sksurv_common import SksurvModelBase


class SRFModel(SksurvModelBase):
    """RandomSurvivalForest (sksurv) has no custom nn.Module of its own --
    see pkgs/models/sksurv_common.py's module docstring for why this and
    gbsa.py/survival_svm.py's classes all subclass one shared base rather
    than duplicating predictions()."""
    pass
