"""DVC stage: download raw UCI Adult data to data/raw/adult.data."""

from synthguard.config import PROJECT_ROOT, load_params
from synthguard.data.download import download

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "adult.data"


def main() -> None:
    d = load_params()["data"]
    actual = download(d["url"], d["member"], d.get("sha256") or "", RAW_PATH)
    if not d.get("sha256"):
        print(f"WARNING: no pinned hash. Pin this in params.yaml data.sha256: {actual}")
    print(f"Saved {RAW_PATH} (sha256={actual})")


if __name__ == "__main__":
    main()
