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

"""Remove blocked PEM roots and reject blocked trust anchors in image files.

Full-image remediation also applies package-specific fixes. Explicit subdirectory
scans only clean PEM stores; --check never modifies files.

Inspect PEM, DER and embedded public keys, ZIP/TAR/ar archives, gzip/bzip2/xz
streams, SQLite BLOBs and PKCS#12 stores. Recognized encrypted or unsupported
containers fail inspection rather than being treated as clean. Virtual runtime
filesystems and CUDA device binaries are excluded; package cache mounts must not
be attached during scanning. Host executables and shared libraries remain in scope.
"""

import argparse
import base64
import bz2
import csv
import gzip
import hashlib
import io
import lzma
import mmap
import os
import re
import shutil
import site
import sqlite3
import ssl
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import BinaryIO

# SHA-256 fingerprints of the DER certificates, not subject-name matches.
_BLOCKED_ROOTS = {
    # emSign Root TLS CA - G3
    "7dd78d5f4f13459a83dff9abbba62edbaf6f2d102bf257fd712f4d9f2746ed8d",
    # emSign Root CS CA - G3
    "f2edaaff3d9b701fb4d14c9f285f8c0bad90fc3bcc1fa07d47f4718edaba34a5",
    # AAA Certificate Services
    "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4",
    # emSign ECC Root CA - G3
    "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b",
}

# Some TLS libraries embed only the root's SubjectPublicKeyInfo sequence contents.
_BLOCKED_SPKI = {
    # emSign Root TLS CA - G3
    "c619e4780ed8e0abeda24ad732567de4383a7ea941338f6d1a7d1b49ac792fe9",
    # emSign Root CS CA - G3
    "0a6c2b2ca5b364207954171987ac9d2178561569d8205457e77ca90c32ede282",
    # AAA Certificate Services
    "bd153ed7b0434f6886b17bce8bbe84ed340c7132d702a8f4fa318f756ecbd6f3",
    # emSign ECC Root CA - G3
    "8d417db2dd8bf5e3084d1e3f196d583849d81bdd4c00c70b9d39369e96b8c782",
}
_SPKI_HEADER = re.compile(
    rb"\x30(?:\x10\x06\x07\x2a\x86\x48\xce\x3d\x02\x01"
    rb"|\x0d\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x01)"
)
_CERTIFICATE = re.compile(
    rb"-----BEGIN (?:TRUSTED )?CERTIFICATE-----[A-Za-z0-9+/=\s]+-----END (?:TRUSTED )?CERTIFICATE-----"
)
# OpenSSL -text descriptions may precede PEM blocks in otherwise plain CA bundles.
_CERTIFICATE_TEXT = re.compile(
    rb"^Certificate:\r?\n.*?(?=^-----BEGIN (?:TRUSTED )?CERTIFICATE-----)", re.MULTILINE | re.DOTALL
)
_CERTIFICATE_LABEL = re.compile(rb"^[^\r\n]+\r?\n=+\r?\n", re.MULTILINE)
# DER SEQUENCE headers of the four certificates, including their encoded lengths.
_DER_HEADER = re.compile(rb"\x30\x82(?:\x02[\x28\x29\x4e]|\x04\x32)")
_VIRTUAL_FILESYSTEMS = {Path("/dev"), Path("/proc"), Path("/sys"), Path("/run")}
_MAX_DEPTH = 12


def _pem_der(pem: bytes) -> bytes:
    normalized = pem.replace(b"TRUSTED CERTIFICATE", b"CERTIFICATE")
    der = ssl.PEM_cert_to_DER_cert(normalized.decode("ascii"))
    # OpenSSL trusted PEMs append X509_AUX data after the certificate's DER SEQUENCE.
    if b"TRUSTED CERTIFICATE" in pem:
        length_bytes = der[1] & 0x7F if der[1] & 0x80 else 0
        size = int.from_bytes(der[2 : 2 + length_bytes], "big") if length_bytes else der[1]
        return der[: 2 + length_bytes + size]
    return der


def _blocked_pem(match: re.Match[bytes]) -> bool:
    der = _pem_der(match[0])
    return hashlib.sha256(der).hexdigest() in _BLOCKED_ROOTS


def _check_public_keys(data: bytes | mmap.mmap, label: str) -> None:
    for match in _SPKI_HEADER.finditer(data):
        start = match.start()
        if data[start + 1] == 0x10:
            der = b"\x30\x76" + data[start : start + 118]
        else:
            der = b"\x30\x82\x01\x22" + data[start : start + 290]
        digest = hashlib.sha256(der).hexdigest()
        if digest in _BLOCKED_SPKI:
            raise RuntimeError(f"{label}: blocked root public key remains (SHA-256 {digest})")


def _check_certificates(data: bytes | mmap.mmap, label: str) -> None:
    for match in _CERTIFICATE.finditer(data):
        if _blocked_pem(match):
            raise RuntimeError(f"{label}: blocked PEM certificate remains")
        _check_public_keys(_pem_der(match[0]), label)
    _check_public_keys(data, label)
    for match in _DER_HEADER.finditer(data):
        start = match.start()
        size = int.from_bytes(data[start + 2 : start + 4], "big") + 4
        if hashlib.sha256(data[start : start + size]).hexdigest() in _BLOCKED_ROOTS:
            raise RuntimeError(f"{label}: blocked DER certificate remains")


def _remove_roots(path: Path, *, check: bool) -> None:
    # Only edit plain PEM stores. A PEM string inside source code or a binary is
    # reported rather than rewritten, since changing it could corrupt the file.
    original = path.read_bytes()
    matches = list(_CERTIFICATE.finditer(original))
    if not any(_blocked_pem(match) for match in matches):
        return
    remainder = _CERTIFICATE_LABEL.sub(
        b"", _CERTIFICATE.sub(b"", _CERTIFICATE_TEXT.sub(b"", original))
    )
    plain_pem = all(
        not line.strip() or line.lstrip().startswith(b"#") for line in remainder.splitlines()
    )
    if check or not plain_pem:
        raise RuntimeError(f"{path}: blocked PEM certificate remains; update the owning package")
    cleaned = _CERTIFICATE.sub(lambda match: b"" if _blocked_pem(match) else match[0], original)
    # An individual blocked root becomes an empty file and can no longer be loaded
    # as a trust anchor. Keep its path and permissions intact for package management.
    certificates = "".join(
        ssl.DER_cert_to_PEM_cert(_pem_der(pem)) for pem in _CERTIFICATE.findall(cleaned)
    )
    if certificates:
        ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cadata=certificates)
    _check_certificates(cleaned, str(path))
    path.write_bytes(cleaned)
    print(f"Removed blocked PEM roots from {path}", flush=True)


def _is_cuda_kernel(header: bytes) -> bool:
    if (
        len(header) >= 20
        and header[:4] == b"\x7fELF"
        and header[4] in (1, 2)
        and header[5] in (1, 2)
        and header[6] == 1
    ):
        byteorder = "little" if header[5] == 1 else "big"
        # EM_CUDA identifies a device ELF, unlike host libraries containing CUDA code.
        return int.from_bytes(header[18:20], byteorder) == 190
    # Standalone fatbin header: magic, version 1, and a 16-byte header.
    return len(header) >= 16 and header[:8] == b"\x50\xed\x55\xba\x01\x00\x10\x00"


def _scan_stream(stream: BinaryIO, label: str, depth: int) -> None:
    header = stream.read(64)
    if _is_cuda_kernel(header):
        return
    # Spool archive members to disk so large libraries do not consume their size in RAM.
    with tempfile.TemporaryFile() as temporary:
        temporary.write(header)
        shutil.copyfileobj(stream, temporary)
        temporary.flush()
        if temporary.tell():
            with mmap.mmap(temporary.fileno(), 0, access=mmap.ACCESS_READ) as data:
                _scan_data(data, label, depth, stream=temporary)


def _scan_sqlite(data: mmap.mmap, label: str, depth: int) -> None:
    # Read logical BLOB values, including certificates split across SQLite pages.
    with tempfile.NamedTemporaryFile() as temporary:
        temporary.write(data)
        temporary.flush()
        connection = sqlite3.connect(f"file:{temporary.name}?mode=ro", uri=True)
        try:
            tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (table,) in tables:
                quoted = table.replace('"', '""')
                for row in connection.execute(f'SELECT * FROM "{quoted}"'):
                    for value in row:
                        if isinstance(value, bytes):
                            _scan_stream(io.BytesIO(value), f"{label}:{table}", depth + 1)
        finally:
            connection.close()


def _is_pkcs12(data: mmap.mmap) -> bool:
    if len(data) < 8 or data[0] != 0x30:
        return False
    length_bytes = data[1] & 0x7F if data[1] & 0x80 else 0
    offset = 2 + length_bytes
    # PFX version 3 followed by an authenticated-safe PKCS#7 data ContentInfo.
    return (
        data[offset : offset + 3] == b"\x02\x01\x03"
        and b"\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x07\x01" in data[:64]
    )


def _is_tar(data: mmap.mmap) -> bool:
    if len(data) < 512:
        return False
    try:
        checksum = int(data[148:156].strip(b"\x00 "), 8)
    except ValueError:
        return False
    return checksum == sum(data[:148]) + 8 * ord(" ") + sum(data[156:512])


def _is_appended_zip(data: mmap.mmap, stream: BinaryIO) -> bool:
    if b"PK\x05\x06" not in data[-65557:]:
        return False
    # Source and bytecode files can contain ZIP signature constants. Require a
    # readable central directory before interpreting a non-ZIP-prefixed file as an archive.
    try:
        with zipfile.ZipFile(stream):
            return True
    except zipfile.BadZipFile:
        return False


def _scan_data(data: mmap.mmap, label: str, depth: int = 0, *, stream: BinaryIO) -> None:
    if depth > _MAX_DEPTH:
        raise RuntimeError(f"{label}: archive nesting exceeds inspection limit")
    if _is_cuda_kernel(data[:64]):
        return
    _check_certificates(data, label)
    stream.seek(0)
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08") or _is_appended_zip(data, stream):
        with zipfile.ZipFile(stream) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    with archive.open(member) as stream:
                        _scan_stream(stream, f"{label}!{member.filename}", depth + 1)
    elif data[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=stream) as stream:
            _scan_stream(stream, f"{label}!gzip", depth + 1)
    elif data[:3] == b"BZh":
        with bz2.BZ2File(stream) as stream:
            _scan_stream(stream, f"{label}!bzip2", depth + 1)
    elif data[:6] == b"\xfd7zXZ\x00":
        with lzma.LZMAFile(stream) as stream:
            _scan_stream(stream, f"{label}!xz", depth + 1)
    elif _is_tar(data):
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            for member in archive:
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise RuntimeError(f"{label}!{member.name}: cannot read archive member")
                    with stream:
                        _scan_stream(stream, f"{label}!{member.name}", depth + 1)
    elif data[:8] == b"!<arch>\n":
        offset = 8
        while offset < len(data):
            header = data[offset : offset + 60]
            if len(header) != 60 or header[-2:] != b"`\n":
                raise RuntimeError(f"{label}: invalid ar archive")
            size = int(header[48:58])
            if offset + 60 + size > len(data):
                raise RuntimeError(f"{label}: truncated ar archive")
            name = header[:16].decode("ascii").strip()
            _scan_stream(
                io.BytesIO(data[offset + 60 : offset + 60 + size]), f"{label}!{name}", depth + 1
            )
            offset += 60 + size + size % 2
    elif data[:16] == b"SQLite format 3\x00":
        _scan_sqlite(data, label, depth)
    elif _is_pkcs12(data) or label.lower().endswith((".p12", ".pfx")):
        result = subprocess.run(
            ["openssl", "pkcs12", "-nokeys", "-passin", "pass:"],
            input=data[:],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"{label}: cannot inspect PKCS#12 store with an empty password")
        _check_certificates(result.stdout, label)
    elif (
        data[:4] == b"\x28\xb5\x2f\xfd"
        or data[:6] == b"7z\xbc\xaf\x27\x1c"
        or data[:4] == b"Rar!"
        or label.lower().endswith(".bcfks")
    ):
        raise RuntimeError(f"{label}: unsupported archive/keystore; cannot verify its certificates")


def _scan_file(path: Path, *, check: bool) -> None:
    with path.open("rb") as stream:
        if not os.fstat(stream.fileno()).st_size or _is_cuda_kernel(stream.read(64)):
            return
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            has_blocked_pem = any(_blocked_pem(match) for match in _CERTIFICATE.finditer(data))
            if not has_blocked_pem:
                _scan_data(data, str(path), stream=stream)
                return
    _remove_roots(path, check=check)
    # Reopen after a possible size change; never truncate a live mmap.
    _scan_file(path, check=True)


def _clean_pip_wheel(path: Path) -> None:
    bundle = "pip/_vendor/certifi/cacert.pem"
    with zipfile.ZipFile(path) as source, tempfile.TemporaryDirectory() as directory:
        pem = Path(directory) / "cacert.pem"
        original = source.read(bundle)
        pem.write_bytes(original)
        _remove_roots(pem, check=False)
        cleaned = pem.read_bytes()
        if cleaned == original:
            return
        record = next(name for name in source.namelist() if name.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(source.read(record).decode())))
        digest = base64.urlsafe_b64encode(hashlib.sha256(cleaned).digest()).rstrip(b"=").decode()
        for row in rows:
            if row[0] == bundle:
                row[1:] = [f"sha256={digest}", str(len(cleaned))]
        output = io.StringIO(newline="")
        csv.writer(output).writerows(rows)
        replacement = Path(directory) / path.name
        with zipfile.ZipFile(replacement, "w") as target:
            for member in source.infolist():
                data = source.read(member)
                if member.filename == bundle:
                    data = cleaned
                elif member.filename == record:
                    data = output.getvalue().encode()
                target.writestr(member, data)
        # Preserve the existing file mode and ownership.
        path.write_bytes(replacement.read_bytes())
    print(f"Cleaned bundled pip trust store: {path}", flush=True)


def _mitigate_packages() -> None:
    for directory in site.getsitepackages():
        packages = Path(directory)
        server = packages / "tensorboard_data_server/bin/server"
        # A missing optional server selects TensorBoard's Python data loader.
        server.unlink(missing_ok=True)
        for library in packages.glob("openai_harmony/*.so"):
            try:
                _check_public_keys(library.read_bytes(), str(library))
            except RuntimeError:
                subprocess.run(
                    ["bash", str(Path(__file__).with_name("rebuild_harmony.sh"))], check=True
                )
                _check_public_keys(library.read_bytes(), str(library))

    wheels = list(Path("/usr/share/python-wheels").glob("pip-*.whl"))
    wheels.extend(
        Path("/usr/local").glob(
            "cuda-*/NsightSystems*/target-*/python/lib/python*/ensurepip/_bundled/pip-*.whl"
        )
    )
    for wheel in wheels:
        _clean_pip_wheel(wheel)
    Path("/opt/pytorch/torch_tensorrt/tests/py/ts/models/hw_compat.ts").unlink(missing_ok=True)


def _raise_walk_error(error: OSError) -> None:
    raise error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check only; do not modify image files"
    )
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("/")])
    args = parser.parse_args()

    system_roots = ssl.create_default_context().get_ca_certs(binary_form=True)
    if not system_roots or any(
        hashlib.sha256(cert).hexdigest() in _BLOCKED_ROOTS for cert in system_roots
    ):
        raise RuntimeError("Update the system ca-certificates package before scanning")

    if not args.check and args.roots == [Path("/")]:
        _mitigate_packages()

    seen = set()
    for root in args.roots:
        for directory, directories, filenames in os.walk(root, onerror=_raise_walk_error):
            directories[:] = [
                name for name in directories if Path(directory) / name not in _VIRTUAL_FILESYSTEMS
            ]
            for filename in filenames:
                path = Path(directory) / filename
                info = path.lstat()
                # Symlink targets are visited at their real locations in the image.
                if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) in seen:
                    continue
                seen.add((info.st_dev, info.st_ino))
                try:
                    _scan_file(path, check=args.check)
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    zipfile.BadZipFile,
                    tarfile.TarError,
                    sqlite3.Error,
                    EOFError,
                    lzma.LZMAError,
                ) as error:
                    raise RuntimeError(f"Certificate scan failed for {path}: {error}") from error
    print(f"Scanned {len(seen)} image files: no blocked roots detected", flush=True)


if __name__ == "__main__":
    main()
