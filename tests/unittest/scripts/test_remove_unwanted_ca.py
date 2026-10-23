# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only regression tests for the release-container certificate scanner."""

import base64
import csv
import gzip
import hashlib
import importlib.util
import io
import re
import sqlite3
import ssl
import struct
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.cpu_only

_SCANNER_PATH = Path(__file__).resolve().parents[3] / "docker/common/remove_unwanted_ca.py"
_SPEC = importlib.util.spec_from_file_location("remove_unwanted_ca", _SCANNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
scanner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scanner)


class CertificateScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "arbitrary-name"
        roots = ssl.create_default_context().get_ca_certs(binary_form=True)
        self.blocked, self.allowed = roots[:2]
        self.pem = ssl.DER_cert_to_PEM_cert(self.blocked).encode("ascii")
        blocked = patch.object(
            scanner, "_BLOCKED_ROOTS", {hashlib.sha256(self.blocked).hexdigest()}
        )
        header = patch.object(scanner, "_DER_HEADER", re.compile(re.escape(self.blocked[:4])))
        blocked.start()
        header.start()
        self.addCleanup(blocked.stop)
        self.addCleanup(header.stop)

    def assert_rejected(self, content: bytes) -> None:
        self.path.write_bytes(content)
        with self.assertRaises(RuntimeError):
            scanner._scan_file(self.path, check=False)
        self.assertEqual(self.path.read_bytes(), content)

    def test_plain_pem_without_known_filename(self) -> None:
        allowed = ssl.DER_cert_to_PEM_cert(self.allowed).encode("ascii")
        self.path.write_bytes(b"# retained comment\n" + self.pem + allowed)
        with self.assertRaises(RuntimeError):
            scanner._scan_file(self.path, check=True)
        scanner._scan_file(self.path, check=False)
        expected = b"# retained comment\n\n" + allowed
        self.assertEqual(self.path.read_bytes(), expected)
        scanner._scan_file(self.path, check=False)
        scanner._scan_file(self.path, check=True)
        self.assertEqual(self.path.read_bytes(), expected)

    def test_package_mitigation_only_for_full_image_remediation(self) -> None:
        for arguments, expected in [([], True), (["--check"], False), (["/tmp"], False)]:
            with (
                self.subTest(arguments=arguments),
                patch("sys.argv", ["remove_unwanted_ca.py", *arguments]),
                patch.object(scanner, "_mitigate_packages") as mitigate,
                patch.object(scanner, "_BLOCKED_ROOTS", set()),
                patch.object(scanner.os, "walk", return_value=[]),
            ):
                scanner.main()
                self.assertEqual(mitigate.called, expected)

    def test_clean_pip_wheel(self) -> None:
        bundle = "pip/_vendor/certifi/cacert.pem"
        record = "pip-24.0.dist-info/RECORD"
        allowed = ssl.DER_cert_to_PEM_cert(self.allowed).encode("ascii")
        with zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(bundle, self.pem + allowed)
            archive.writestr("pip/__init__.py", b"unchanged")
            archive.writestr(record, f"{bundle},sha256=old,1\n{record},,\n")
        self.path.chmod(0o640)
        scanner._clean_pip_wheel(self.path)
        with zipfile.ZipFile(self.path) as archive:
            self.assertIsNone(archive.testzip())
            cleaned = archive.read(bundle)
            self.assertIn(allowed, cleaned)
            self.assertNotIn(self.pem, cleaned)
            self.assertEqual(archive.read("pip/__init__.py"), b"unchanged")
            rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
            digest = base64.urlsafe_b64encode(hashlib.sha256(cleaned).digest()).rstrip(b"=")
            self.assertEqual(rows[0], [bundle, "sha256=" + digest.decode(), str(len(cleaned))])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o640)
        scanner._scan_file(self.path, check=True)
        original = self.path.read_bytes()
        scanner._clean_pip_wheel(self.path)
        self.assertEqual(self.path.read_bytes(), original)

    def test_individual_pem(self) -> None:
        self.path.write_bytes(self.pem)
        scanner._scan_file(self.path, check=False)
        self.assertFalse(self.path.read_bytes().strip())
        scanner._scan_file(self.path, check=True)

    def test_openssl_text_pem_bundle(self) -> None:
        described = subprocess.run(
            ["openssl", "x509", "-text"], input=self.pem, capture_output=True, check=True
        ).stdout
        self.path.write_bytes(described)
        scanner._scan_file(self.path, check=False)
        scanner._scan_file(self.path, check=True)
        self.assertNotIn(b"-----BEGIN CERTIFICATE-----", self.path.read_bytes())

    def test_labeled_pem_bundle(self) -> None:
        self.path.write_bytes(b"Example Root CA\n===============\n" + self.pem)
        scanner._scan_file(self.path, check=False)
        scanner._scan_file(self.path, check=True)
        self.assertNotIn(b"-----BEGIN CERTIFICATE-----", self.path.read_bytes())

    def test_openssl_trusted_pem(self) -> None:
        trusted = subprocess.run(
            ["openssl", "x509", "-addtrust", "serverAuth", "-trustout"],
            input=self.pem,
            capture_output=True,
            check=True,
        ).stdout
        allowed = ssl.DER_cert_to_PEM_cert(self.allowed).encode("ascii")
        self.path.write_bytes(trusted + allowed)
        scanner._scan_file(self.path, check=False)
        scanner._scan_file(self.path, check=True)
        self.assertEqual(
            scanner._CERTIFICATE.findall(self.path.read_bytes()),
            scanner._CERTIFICATE.findall(allowed),
        )

    def test_standalone_der(self) -> None:
        self.assert_rejected(self.blocked)

    def test_binary_der(self) -> None:
        self.assert_rejected(b"\x7fELF\x00prefix" + self.blocked + b"suffix")

    def test_binary_pem(self) -> None:
        self.assert_rejected(b"\x7fELF\x00prefix" + self.pem + b"suffix")

    def test_source_string_is_not_rewritten(self) -> None:
        self.assert_rejected(b'CERT = """' + self.pem + b'"""')

    @staticmethod
    def elf_header(machine: int) -> bytes:
        header = bytearray(64)
        header[:7] = b"\x7fELF\x02\x01\x01"
        header[18:20] = machine.to_bytes(2, "little")
        return bytes(header)

    def test_cuda_device_binary_is_skipped(self) -> None:
        content = self.elf_header(190) + self.blocked
        self.path.write_bytes(content)
        scanner._scan_file(self.path, check=True)
        self.assertEqual(self.path.read_bytes(), content)

    def test_fatbin_is_skipped(self) -> None:
        content = struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(self.blocked)) + self.blocked
        self.path.write_bytes(content)
        scanner._scan_file(self.path, check=True)
        self.assertEqual(self.path.read_bytes(), content)

    def test_host_binary_with_kernel_filename_is_scanned(self) -> None:
        self.path = self.path.with_suffix(".cubin")
        self.assert_rejected(self.elf_header(62) + self.blocked)

    def test_archived_cuda_kernel_is_skipped(self) -> None:
        with zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("device", self.elf_header(190) + self.blocked)
        scanner._scan_file(self.path, check=True)

    def test_embedded_public_key_without_certificate(self) -> None:
        key = subprocess.run(
            ["openssl", "genpkey", "-algorithm", "EC", "-pkeyopt", "ec_paramgen_curve:secp384r1"],
            check=True,
            capture_output=True,
        ).stdout
        public = subprocess.run(
            ["openssl", "pkey", "-pubout", "-outform", "DER"],
            input=key,
            check=True,
            capture_output=True,
        ).stdout
        with patch.object(scanner, "_BLOCKED_SPKI", {hashlib.sha256(public).hexdigest()}):
            for representation in (public, public[2:]):
                with self.subTest(size=len(representation)):
                    self.assert_rejected(b"binary-prefix" + representation + b"suffix")

    def test_nested_compressed_archive(self) -> None:
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
            entry = tarfile.TarInfo("renamed-certificate")
            entry.size = len(self.pem)
            archive.addfile(entry, io.BytesIO(self.pem))
        wheel = io.BytesIO()
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("payload", gzip.compress(archive_bytes.getvalue()))
        self.assert_rejected(wheel.getvalue())

    def test_archive_appended_to_executable(self) -> None:
        archive_bytes = io.BytesIO(b"executable-prefix")
        with zipfile.ZipFile(archive_bytes, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("cert", self.pem)
        self.assert_rejected(archive_bytes.getvalue())

    def test_zip_signature_constant_is_not_an_archive(self) -> None:
        footer = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, 100, 999999, 0)
        self.path.write_bytes(b"bytecode-prefix" + footer)
        scanner._scan_file(self.path, check=True)

    def test_unrelated_archive(self) -> None:
        with zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("certificate", ssl.DER_cert_to_PEM_cert(self.allowed))
        scanner._scan_file(self.path, check=True)

    def test_sqlite_overflow_pages(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA page_size=512")
            connection.execute("CREATE TABLE certificates (value BLOB)")
            connection.execute("INSERT INTO certificates VALUES (?)", (self.blocked,))
        with self.assertRaises(RuntimeError):
            scanner._scan_file(self.path, check=True)

    def test_pkcs12_without_extension(self) -> None:
        source = Path(self.directory.name) / "input.pem"
        source.write_bytes(self.pem)
        subprocess.run(
            [
                "openssl",
                "pkcs12",
                "-export",
                "-nokeys",
                "-in",
                str(source),
                "-out",
                str(self.path),
                "-passout",
                "pass:",
            ],
            check=True,
            capture_output=True,
        )
        with self.assertRaises(RuntimeError):
            scanner._scan_file(self.path, check=True)

    def test_password_protected_pkcs12_fails_closed(self) -> None:
        source = Path(self.directory.name) / "input.pem"
        source.write_bytes(ssl.DER_cert_to_PEM_cert(self.allowed).encode("ascii"))
        subprocess.run(
            [
                "openssl",
                "pkcs12",
                "-export",
                "-nokeys",
                "-in",
                str(source),
                "-out",
                str(self.path),
                "-passout",
                "pass:test",
            ],
            check=True,
            capture_output=True,
        )
        with self.assertRaisesRegex(RuntimeError, "cannot inspect PKCS#12"):
            scanner._scan_file(self.path, check=True)


if __name__ == "__main__":
    unittest.main()
