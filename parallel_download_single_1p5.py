"""Compatibility entry point for the maintained data/ downloader."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    scripts = Path(__file__).resolve().parent / 'data'
    sys.path.insert(0, str(scripts))
    runpy.run_path(str(scripts / Path(__file__).name), run_name='__main__')
