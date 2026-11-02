"""Run the real GPU/data acceptance sequence, keeping every step's output.

This does not allocate a runtime, mount Drive, or publish anything. Default:
one training epoch followed by a checkpoint resume to epoch two, both using
the unchanged 100-epoch schedule. Final ablations remain separate runs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def verify_source(root):
    manifest_path = root / "colab_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Invalid packaged source manifest")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or out-of-root packaged source: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Packaged source changed: {relative}")
    return {"sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "files": manifest["files"]}


def require_complete_pytest(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if not cases or any(case.find("skipped") is not None for case in cases):
        raise ValueError("GPU acceptance requires the full Torch suite, without skipped tests")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new acceptance-log directory")
    parser.add_argument("--train-out", type=Path, required=True, help="new training directory; keep its best.pt and last.pt together")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--skip-resume-check", action="store_true", help="stop at epoch one instead of also testing resume to epoch two")
    args = parser.parse_args()
    out, train_out, data = args.out.resolve(), args.train_out.resolve(), args.data_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("acceptance output is not empty; choose a fresh directory")
    if train_out.exists() and any(train_out.iterdir()):
        parser.error("training output is not empty; resume train.py directly in that directory")
    if out == train_out or out in train_out.parents or train_out in out.parents:
        parser.error("acceptance logs and training output must be separate sibling directories")
    out.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 1, "status": "running", "started_utc": now(),
             "project_root": str(ROOT), "data_dir": str(data), "train_out": str(train_out),
             "experiment_complete": False, "steps": []}
    state_path = out / "acceptance.json"
    write_json(state_path, state)

    def run(name, command):
        record = {"name": name, "command": command, "started_utc": now(), "status": "running"}
        state["steps"].append(record)
        write_json(state_path, state)
        print(f"[{now()}] {name}", flush=True)
        began = time.monotonic()
        log = out / f"{len(state['steps']):02d}_{name}.log"
        record["log"] = log.name
        try:
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True,
                                           encoding="utf-8", errors="replace", bufsize=1)
                try:
                    for line in process.stdout:
                        stream.write(line)
                        stream.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            record["exit_code"] = code
            if code:
                raise subprocess.CalledProcessError(code, command)
            record["status"] = "ok"
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - began
            record["finished_utc"] = now()
            write_json(state_path, state)

    py = sys.executable
    try:
        state["source"] = verify_source(ROOT)
        if args.install_deps:
            run("install", [py, "-m", "pip", "install", "-r", "requirements-test.txt"])
        run("gpu_environment", [py, "-c", "import json, torch; assert torch.cuda.is_available(), 'CUDA is required'; print(json.dumps({'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}))"])
        run("nvidia_smi", ["nvidia-smi"])
        run("tests", [py, "-m", "pytest", "tests", "-q", "-ra", f"--junitxml={out / 'pytest.xml'}"])
        state["tests_executed"] = require_complete_pytest(out / "pytest.xml")
        run("gpu_smoke", [py, "gpu_smoke.py", "--out", str(out / "gpu_smoke.json")])
        run("voc_download", [py, "download_voc.py", "--data-dir", str(data)])
        manifests = [str(data / "manifests" / f"{split}.json") for split in ("train", "val", "test")]
        run("preflight", [py, "preflight.py", "--manifests", *manifests,
                          "--voc-root", str(data / "VOCdevkit"), "--out", str(out / "preflight.json")])
        base = [py, "train.py", "--voc-root", str(data / "VOCdevkit"),
                "--train-manifest", manifests[0], "--val-manifest", manifests[1],
                "--out", str(train_out)]
        run("epoch_one", [*base, "--stop-after-epoch", "1"])
        if not args.skip_resume_check:
            run("resume_epoch_two", [*base, "--resume", str(train_out / "last.pt"), "--stop-after-epoch", "2"])
        training = json.loads((train_out / "run.json").read_text(encoding="utf-8"))
        expected_epochs = 1 if args.skip_resume_check else 2
        if training["completed_epochs"] != expected_epochs or training["status"] != "paused_at_epoch_boundary":
            raise ValueError("Training did not pause at the expected acceptance boundary")
        state.update(status="passed", completed_epochs=expected_epochs,
                     continuation_command=[*base, "--resume", str(train_out / "last.pt")])
        print("GPU/data acceptance passed. Full 100-epoch training remains to be completed.", flush=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["finished_utc"] = now()
        write_json(state_path, state)


if __name__ == "__main__":
    main()
