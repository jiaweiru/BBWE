from .valentini_prepare import prepare_vltn
from .vctk_prepare import prepare_vctk


def prepare_dataset(corpus, *args, **kargs):
    assert corpus in [
        "VCTK",
        "Valentini",
    ], f"The only datasets available are VCTK and Valentini, not {corpus}"
    if corpus == "VCTK":
        prepare_vctk(*args, **kargs)
    elif corpus == "Valentini":
        prepare_vltn(*args, **kargs)
