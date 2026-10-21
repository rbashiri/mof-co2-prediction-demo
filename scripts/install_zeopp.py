"""Download, verify and compile Zeo++ 0.3 using Python 3.12+ on Linux."""

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


SOURCE_URL = "https://www.zeoplusplus.org/zeo++-0.3.tar.gz"
SOURCE_SHA256 = "a3db73ad8a1a9ded6899d86fce5297b7c62e4ed88c8fa6a3e887b71daba9faac"


def install(destination: Path) -> None:
    for tool in ("make", "g++"):
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"Required build tool is missing: {tool}")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. "
            "Choose a new --destination; existing installations are not overwritten."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="zeopp-build-", dir=destination.parent
    ) as directory:
        work = Path(directory)
        archive = work / "zeopp-0.3.tar.gz"
        print(f"Downloading {SOURCE_URL}", flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            with archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != SOURCE_SHA256:
            raise ValueError(
                f"Source checksum mismatch: expected {SOURCE_SHA256}, got {digest}"
            )

        with tarfile.open(archive, "r:gz") as source:
            source.extractall(work, filter="data")
        source_dir = work / "zeo++-0.3"
        for build_dir in (source_dir / "voro++" / "src", source_dir):
            print(f"Building in {build_dir}", flush=True)
            subprocess.run(["make"], cwd=build_dir, check=True, timeout=1800)

        subprocess.run(
            [str(source_dir / "network")],
            cwd=source_dir,
            check=True,
            timeout=30,
        )
        shutil.move(str(source_dir), str(destination))

    print(f"\nZeo++ installed and executable verified: {destination / 'network'}")
    print("Geometry extraction is not yet connected to the application.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path.home() / "zeopp-tools" / "zeo++-0.3",
        help="New installation directory (existing directories are never overwritten).",
    )
    args = parser.parse_args()
    try:
        install(args.destination.expanduser().resolve())
    except (
        OSError,
        ValueError,
        tarfile.TarError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        parser.exit(1, f"Zeo++ installation failed: {exc}\n")


if __name__ == "__main__":
    main()
