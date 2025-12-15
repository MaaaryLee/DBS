"""
Compatibility shim.

Some notebooks/scripts import `BGN_MC_Online` from the repository root. The actual
implementation lives in `core/BGN_MC_Online.py`.
"""

from core.BGN_MC_Online import BGN_MC_Online  # noqa: F401


