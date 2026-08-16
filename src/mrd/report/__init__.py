"""Report assembly and rendering."""

from .html import render, sparkline, write
from .model import Attempt, CaseDiff, ReportData, build

__all__ = ["Attempt", "CaseDiff", "ReportData", "build", "render", "sparkline", "write"]
