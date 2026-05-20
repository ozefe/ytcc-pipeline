"""Pipeline orchestrator + per-stage helpers.

Public surface: `process_pdf`. Everything else under this package is internal -- module
names indicate scope. The formula stage proper (`run_formula_stage`) lives under the
processors layer because it owns its own dataclass + model wrapper.
"""

from .orchestrator import process_pdf

__all__ = ["process_pdf"]
