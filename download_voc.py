"""Download verified VOC2007 archives and extract only files within data/VOCdevkit."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.error
import urllib.request

from prepare_voc import prepare
from preflight import check

# Published by torchvision's official VOC dataset implementation.
# MD5 checks archive identity/corruption; HTTPS authenticates the transfer.
ARCHIVES = {
    "VOCtrainval_06-Nov-2007.tar": "c52e279531787c972589f7e41ab4ae64",
    "VOCtest_06-Nov-2007.tar": "b6e924de25625d8de591ea690078ad9f",
}
BASE_URL = "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/"
MIRROR_COMMIT = "dd873d3cf7b375ce99012a6c4d64729ab0d76602"
BASE_URLS = (
    BASE_URL,
    f"https://huggingface.co/datasets/DerrickUnleashed/Pascal_VOC/resolve/{MIRROR_COMMIT}/",
)
# Additional byte identities from the pinned mirror's LFS objects. The official
# torchvision MD5 values above remain mandatory regardless of download host.
EXPECTED_SHA256 = {
    "VOCtrainval_06-Nov-2007.tar": "7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508",
    "VOCtest_06-Nov-2007.tar": "6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892",
}


def digest(path, algorithm):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def extract_archive(archive, destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:") as stream:
        members = stream.getmembers()
        # Check the entire archive before writing any member.
        for member in members:
            path = PurePosixPath(member.name)
            target = (root / Path(*path.parts)).resolve()
            if (path.is_absolute() or not path.parts or path.parts[0] != "VOCdevkit" or
                    ".." in path.parts or "\\" in member.name or ":" in member.name or
                    not target.is_relative_to(root) or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe archive member: {member.name}")
        for member in members:
            target = root / Path(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def fetch_archive(path, expected_md5, expected_sha256):
    """Try documented sources, accepting only the same frozen archive bytes."""
    attempts = []
    partial = path.with_suffix(".tar.partial")
    for base in BASE_URLS:
        url = base + path.name
        print(f"Downloading {url}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=20) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            attempts.append({"url": url, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"Source unavailable: {type(exc).__name__}; trying the next documented mirror", flush=True)
            continue
        if digest(partial, "md5") != expected_md5 or digest(partial, "sha256") != expected_sha256:
            raise ValueError(f"Downloaded archive checksum mismatch: {partial}")
        partial.replace(path)
        attempts.append({"url": url, "status": "ok"})
        return {"url": url, "source": "download", "attempts": attempts}
    raise RuntimeError(f"All VOC sources failed: {json.dumps(attempts)}")


def download(data_dir):
    data_dir = Path(data_dir)
    archive_dir = data_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = archive_dir / "download_receipt.json"
    previous = {r["file"]: r for r in json.loads(receipt_path.read_text())} if receipt_path.exists() else {}
    receipts = []
    for name, expected in ARCHIVES.items():
        path = archive_dir / name
        if path.exists():
            origin = {k: previous[name][k] for k in ("url", "source", "attempts") if k in previous.get(name, {})}
            origin.setdefault("url", None)
            origin.setdefault("source", "preexisting_verified_cache")
            origin["cache_reused"] = True
        else:
            origin = fetch_archive(path, expected, EXPECTED_SHA256[name])
        actual_sha = digest(path, "sha256")
        if digest(path, "md5") != expected or actual_sha != EXPECTED_SHA256[name]:
            raise ValueError(f"Cached archive checksum mismatch: {path}")
        receipts.append({"file": name, **origin, "expected_md5": expected,
                         "sha256": actual_sha, "expected_sha256": EXPECTED_SHA256[name]})
        # Preserve provenance even when a later archive/source is unavailable.
        receipt_path.write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
        extract_archive(path, data_dir)
    out = data_dir / "manifests"
    audits = prepare(data_dir / "VOCdevkit", "2007", ["train", "val", "test"], out)
    (out / "grid_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    verified = check([out / f"{split}.json" for split in ("train", "val", "test")], data_dir / "VOCdevkit")
    (out / "preflight.json").write_text(json.dumps(verified, indent=2) + "\n")
    print(f"VOC2007 prepared in {data_dir}; all images and XML annotations preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    download(parser.parse_args().data_dir)
