from pathlib import Path
from typing import List

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def get_sample_image_paths(project_root: Path | None = None) -> List[Path]:
    """Return bundled sample image paths sorted by filename."""
    root = project_root or Path(__file__).resolve().parents[1]
    sample_dir = root / "sample_images"
    if not sample_dir.exists():
        return []

    sample_files = [
        path for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(sample_files, key=lambda path: path.name.lower())
