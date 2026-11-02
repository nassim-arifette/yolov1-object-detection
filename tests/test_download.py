import io
import tarfile
from pathlib import Path
import pytest
from download_voc import extract_archive


def test_mirror_uses_same_bytes_and_records_failed_source(tmp_path, monkeypatch):
    import hashlib
    import urllib.error
    import download_voc
    content = b"same frozen archive bytes"
    monkeypatch.setattr(download_voc, "BASE_URLS", ("https://primary/", "https://mirror/"))
    def open_url(url, **kwargs):
        if "primary" in url:
            raise urllib.error.URLError("test timeout")
        return io.BytesIO(content)
    monkeypatch.setattr(download_voc.urllib.request, "urlopen", open_url)
    target = tmp_path / "fixture.tar"
    receipt = download_voc.fetch_archive(target, hashlib.md5(content).hexdigest(), hashlib.sha256(content).hexdigest())
    assert target.read_bytes() == content
    assert receipt["url"] == "https://mirror/fixture.tar"
    assert [row["status"] for row in receipt["attempts"]] == ["failed", "ok"]


def test_wrong_mirror_bytes_are_never_accepted(tmp_path, monkeypatch):
    import download_voc
    monkeypatch.setattr(download_voc, "BASE_URLS", ("https://mirror/",))
    monkeypatch.setattr(download_voc.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong bytes"))
    target = tmp_path / "fixture.tar"
    with pytest.raises(ValueError, match="checksum"):
        download_voc.fetch_archive(target, "0" * 32, "0" * 64)
    assert not target.exists()


def archive(tmp_path, entries):
    path = tmp_path / "fixture.tar"
    with tarfile.open(path, "w") as stream:
        for name, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == "file":
                content = b"fixture"
                member.size = len(content)
                stream.addfile(member, io.BytesIO(content))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = "outside"
                stream.addfile(member)
    return path


def test_valid_file_is_extracted_under_vocdevkit(tmp_path):
    path = archive(tmp_path, [("VOCdevkit/VOC2007/file.txt", "file")])
    out = tmp_path / "data"
    extract_archive(path, out)
    assert (out / "VOCdevkit/VOC2007/file.txt").read_text() == "fixture"


@pytest.mark.parametrize("name,kind", [
    ("../outside.txt", "file"), ("VOCdevkit/../outside.txt", "file"),
    ("/absolute.txt", "file"), ("VOCdevkit/link", "link"),
    ("VOCdevkit/C:/outside.txt", "file"), ("VOCdevkit/..\\outside.txt", "file")])
def test_unsafe_archive_rejected_before_any_member_is_written(tmp_path, name, kind):
    path = archive(tmp_path, [("VOCdevkit/first.txt", "file"), (name, kind)])
    out = tmp_path / "data"
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(path, out)
    assert not list(out.rglob("*.txt"))
