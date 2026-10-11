"""Create a code-only ZIP, including required dependency code and metadata."""
from pathlib import Path
import zipfile

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
OUTPUT = PACKAGE/"EMB_Code_Upload.zip"
IGNORED = {".git", "__pycache__", ".ipynb_checkpoints", ".cache"}
RUNTIME_ROOTS = {"exp", "data", "models", "outputs", ".venv", ".cloud_checks"}
EXTENSIONS = {".py", ".sh", ".yaml", ".yml", ".md", ".json", ".csv", ".txt", ".ipynb"}


def build():
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(REPO.rglob("*")):
            relative = path.relative_to(REPO)
            if (not path.is_file() or path.is_symlink() or relative.parts[0] in RUNTIME_ROOTS
                    or any(part in IGNORED for part in relative.parts)
                    or (path.suffix.lower() not in EXTENSIONS and path.name != "LICENSE")):
                continue
            if not relative.as_posix().isascii():
                raise ValueError(f"Upload filenames must be ASCII: {relative}")
            archive.write(path, arcname="wesep-real-tse/"+relative.as_posix())
    print(f"Code upload: {OUTPUT} ({OUTPUT.stat().st_size/10**6:.2f} MB)")
    return OUTPUT


if __name__ == "__main__":
    build()
