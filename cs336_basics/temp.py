import py7zr
from pathlib import Path

def extract_7z_txt(archive_path, out_dir):
    archive_path = Path(archive_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=out_dir)

    print(f"Extracted to: {out_dir}")

if __name__ == "__main__":
    extract_7z_txt(
        "/home/nipporita/大模型/Week 1/lfs-data/owt_train.7z",
        "/home/nipporita/大模型/Week 1/lfs-data/owt_train"
    )