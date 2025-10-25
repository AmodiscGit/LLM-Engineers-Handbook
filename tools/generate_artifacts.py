#!/usr/bin/env python3
"""Compatibility shim for generate_artifacts.

The real implementation lives at tools/workflow/generate_artifacts.py. Keep this
thin shim so external callers that run "python tools/generate_artifacts.py"
continue to work while we consolidate code under tools.workflow.

This file simply delegates to the new module and shows a small deprecation
message.
"""

from __future__ import annotations
import sys


def main():
    try:
        # delegate to the canonical module
        from tools.workflow.generate_artifacts import main as workflow_main
    except Exception as e:
        print("Failed to import tools.workflow.generate_artifacts:", e, file=sys.stderr)
        print("Please ensure tools/workflow/generate_artifacts.py exists and is importable.")
        raise

    print("NOTE: tools/generate_artifacts.py is deprecated. Delegating to tools.workflow.generate_artifacts.main()")
    return workflow_main()


if __name__ == "__main__":
    raise SystemExit(main())
