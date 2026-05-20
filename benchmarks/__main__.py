"""Allow `python -m benchmarks` to invoke the full sweep runner."""

from benchmarks.run_all import main

raise SystemExit(main())
