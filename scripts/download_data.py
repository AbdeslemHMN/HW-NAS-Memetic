import argparse
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

_NATS_SIMPLE_NAME   = "NATS-tss-v1_0-3ffb9-simple"
# Direct Google Drive file ID for NATS-tss-v1_0-3ffb9-simple.tar (~300 MB)
# Retrieved from the official NATS-Bench GDrive folder:
#   https://drive.google.com/drive/folders/1zjB6wMANiKwB2A1yil2hQ8H_qyeSe2yt
_NATS_GDRIVE_FILE_ID = "17_saCsj_krKjlCBLOJEpNtzPXArMCqxU"
_NATS_GDRIVE_FOLDER  = "https://drive.google.com/drive/folders/1zjB6wMANiKwB2A1yil2hQ8H_qyeSe2yt"
_NATS_ONEDRIVE       = "https://1drv.ms/u/s!Aqkc27lrowWDf6SvuIkSXx0UQaI?e=nfvM5r"


def setup_data():
    """Download HW-NAS-Bench hardware metrics pickle."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    hw_pickle = os.path.join(data_dir, "HW-NAS-Bench-v1_0.pickle")
    if not os.path.exists(hw_pickle):
        hw_repo_url = "https://github.com/GATECH-EIC/HW-NAS-Bench.git"
        temp_repo = os.path.join(project_root, "temp_hw_nas")
        print("Downloading HW-NAS-Bench metrics (shallow clone)...")
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", hw_repo_url, temp_repo],
                check=True,
            )
            shutil.move(os.path.join(temp_repo, "HW-NAS-Bench-v1_0.pickle"), hw_pickle)
            print("Successfully saved HW-NAS-Bench-v1_0.pickle")
        except subprocess.CalledProcessError as e:
            print(f"Error: git clone failed: {e}")
        except FileNotFoundError:
            print("Error: HW-NAS-Bench-v1_0.pickle not found in cloned repo.")
        finally:
            if os.path.exists(temp_repo):
                shutil.rmtree(temp_repo)
    else:
        print("HW-NAS-Bench-v1_0.pickle already exists.")


def download_nats_bench(data_dir: Path) -> None:
    """
    Download and extract NATS-tss-v1_0-3ffb9-simple.tar (~300 MB).
    Uses Dropbox as primary mirror, falls back to manual instructions.

    After extraction:
        data/NATS-tss-v1_0-3ffb9-simple/   (15,625 individual arch files)
    Then run:
        python scripts/extract_nas201_accuracy.py
    """
    dest_dir = data_dir / _NATS_SIMPLE_NAME
    if dest_dir.exists():
        print(f"Already extracted: {dest_dir}")
        print("Run:  python scripts/extract_nas201_accuracy.py")
        return

    tar_path = data_dir / "NATS-tss-v1_0-3ffb9-simple.tar"

    if not tar_path.exists():
        print(f"Downloading NATS-Bench TSS simple archive (~300 MB) …")
        print(f"  Source: Google Drive (gdown)")
        try:
            import gdown  # type: ignore
        except ImportError:
            print("  gdown not installed.  Run:  pip install gdown")
            print(f"  Manual download: https://drive.google.com/file/d/{_NATS_GDRIVE_FILE_ID}")
            return
        try:
            gdown.download(id=_NATS_GDRIVE_FILE_ID, output=str(tar_path), quiet=False)
            print(f"  Saved → {tar_path}")
        except Exception as exc:
            print(f"\n  Download failed: {exc}")
            print("\nManual download options:")
            print(f"  Google Drive (direct) : https://drive.google.com/file/d/{_NATS_GDRIVE_FILE_ID}/view")
            print(f"  Google Drive (folder) : {_NATS_GDRIVE_FOLDER}")
            print(f"  OneDrive              : {_NATS_ONEDRIVE}")
            print(f"  File to download      : NATS-tss-v1_0-3ffb9-simple.tar")
            print(f"  Save to               : {tar_path}")
            print(f"  Then run              : tar xf {tar_path} -C {data_dir}")
            return

    print(f"Extracting {tar_path} …")
    with tarfile.open(str(tar_path)) as tf:
        tf.extractall(str(data_dir))
    print(f"  Extracted → {dest_dir}")
    tar_path.unlink()
    print(f"  Deleted tar (recovered {300} MB).")
    print()
    print("Next step:")
    print("  pip install nats_bench")
    print("  python scripts/extract_nas201_accuracy.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download benchmark data files.")
    parser.add_argument(
        "--nats-bench", action="store_true",
        help=(
            "Download NATS-Bench TSS simple archive (~300 MB) instead of "
            "the 4.7 GB NAS-Bench-201 .pth file. Recommended for most users."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir     = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    setup_data()

    if args.nats_bench:
        download_nats_bench(data_dir)
    else:
        print()
        print("TIP: To get accuracy data, run:")
        print("  python scripts/download_data.py --nats-bench   # ~300 MB")
        print("instead of downloading the 4.7 GB NAS-Bench-201 .pth file.")
