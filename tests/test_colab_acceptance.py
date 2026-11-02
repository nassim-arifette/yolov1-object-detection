"""Acceptance must not relabel changed code or a skipped suite as GPU-verified."""
import hashlib
import json
import pytest

from tools.colab_acceptance import require_complete_pytest, verify_source


def test_changed_packaged_source_is_rejected(tmp_path):
    code = tmp_path / "model.py"
    code.write_text("original source\n")
    manifest = {"schema_version": 1, "files": {
        "model.py": hashlib.sha256(code.read_bytes()).hexdigest()}}
    (tmp_path / "colab_source_manifest.json").write_text(json.dumps(manifest))
    assert verify_source(tmp_path)["files"] == manifest["files"]
    code.write_text("changed source\n")
    with pytest.raises(ValueError, match="changed"):
        verify_source(tmp_path)


def test_source_manifest_cannot_reference_outside_project(tmp_path):
    (tmp_path / "colab_source_manifest.json").write_text(json.dumps({
        "schema_version": 1, "files": {"../private.txt": "0" * 64}}))
    with pytest.raises(ValueError, match="out-of-root"):
        verify_source(tmp_path)


def test_skipped_torch_suite_is_not_acceptance(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text('<testsuites><testsuite><testcase name="torch"><skipped/></testcase></testsuite></testsuites>')
    with pytest.raises(ValueError, match="without skipped"):
        require_complete_pytest(report)
    report.write_text('<testsuites><testsuite><testcase name="loss"/></testsuite></testsuites>')
    assert require_complete_pytest(report) == 1
