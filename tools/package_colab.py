"""Package the current standalone source without publishing it to GitHub."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    files = list(ROOT.glob("*.py"))
    files += [ROOT / name for name in ("README.md", "requirements.txt", "requirements-test.txt")]
    for directory in ("configs", "tests", "tools", "docs"):
        files += [p for p in (ROOT / directory).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    files = sorted(set(files))
    manifest = {"schema_version": 1, "files": {
        p.relative_to(ROOT).as_posix(): digest(p) for p in files}}
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    out = ROOT / "colab_artifacts"
    out.mkdir(exist_ok=True)
    archive = out / "yolov1_colab.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for p in files:
            tf.add(p, arcname=p.relative_to(ROOT).as_posix(), recursive=False)
        info = tarfile.TarInfo("colab_source_manifest.json")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    receipt = {"archive": archive.name, "archive_sha256": digest(archive), **manifest}
    (out / "package_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(archive)
    print("SHA256", receipt["archive_sha256"])
    print(f"{len(files)} source/document files; no data, weights or outputs")


if __name__ == "__main__":
    build()
