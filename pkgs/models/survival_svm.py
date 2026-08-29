from pkgs.models.sksurv_common import SksurvModelBase


class SurvivalSVMModel(SksurvModelBase):
    """FastSurvivalSVM (sksurv) has no custom nn.Module of its own -- see
    pkgs/models/sksurv_common.py's module docstring for why this and
    gbsa.py/srf.py's classes all subclass one shared base rather than
    duplicating predictions()."""
    pass
