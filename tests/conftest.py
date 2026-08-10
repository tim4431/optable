"""Shared test configuration: force a headless matplotlib backend."""

import matplotlib

matplotlib.use("Agg", force=True)
