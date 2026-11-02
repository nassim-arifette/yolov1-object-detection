"""Compatibility notice for the historical, lossy XML-to-text preparation script.

Use prepare_voc.py to preserve difficult flags and every evaluation object.
Importing this module performs no filesystem operations.
"""

if __name__ == "__main__":
    raise SystemExit("Use: python prepare_voc.py --voc-root data/VOCdevkit --out data/manifests")
