"""
Compatibility shim.

Some notebooks/scripts import `BGN_MC` from the repository root. The actual
implementation lives in `core/BGN_MC.py`.
"""

from core.BGN_MC import BGN_MC  # noqa: F401


