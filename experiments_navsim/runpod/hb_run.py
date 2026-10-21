"""Run any pipeline script with the HeavyBridge patch active first.
Usage: python hb_run.py <script.py> [args...]"""
import os, sys, runpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_patch  # noqa: F401
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
