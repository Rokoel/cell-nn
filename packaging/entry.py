"""One executable serves the GUI and its isolated inference subprocesses."""
from fish_viewer.standalone import main

if __name__ == '__main__':
    raise SystemExit(main())
