"""Minimal backtest helpers for the reproduction package.

Only ``backtest.utils`` (repository paths and time helpers) is needed to
reproduce the paper; every script and test in this package imports that module
and nothing else from here.

``backtest.download`` is retained for optional raw-data acquisition and is
intentionally not imported here, so the core import path stays free of heavy
async dependencies.

``backtest.cli`` was removed from the reproduction package. It drove the C++
``_backtest_engine`` extension, which lives in the private development
repository and is not required to reproduce any published value. In this
package it could not run at all: it searched for build artefacts under
``backtest_project/``, a directory that does not exist here, and its failure
message told the reader to run a build command in that directory. Shipping
unreachable code with impossible instructions costs review attention without
buying reproducibility.
"""
