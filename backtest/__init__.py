"""Minimal backtest helpers for the reproduction package.

Only ``backtest.utils`` (repository paths and time helpers) is needed to
reproduce the paper. ``backtest.cli`` and ``backtest.download`` remain in the
package for optional data acquisition but are intentionally not imported here,
so the core import path stays free of heavy async/data dependencies.
"""
