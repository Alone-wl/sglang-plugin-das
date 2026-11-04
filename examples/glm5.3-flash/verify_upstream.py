import hashlib
import json
from pathlib import Path
import sys


def main():
    source = Path(sys.argv[1]).resolve()
    package = source / 'python' / 'sglang'
    if not package.is_dir():
        package = source / 'sglang'
    if not package.is_dir():
        raise SystemExit('SGLANG_DIR must contain python/sglang or sglang')
    manifest = json.loads(Path(__file__).with_name('upstream-source.json').read_text())
    expected = manifest['sha256']
    actual = {
        str(path.relative_to(package)): path
        for path in package.rglob('*')
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'
    }
    changed = [
        name for name, digest in expected.items()
        if name not in actual
        or hashlib.sha256(actual[name].read_bytes()).hexdigest() != digest
    ]
    extra = sorted(set(actual) - set(expected))
    if changed or extra:
        raise SystemExit(json.dumps({'source_mismatch': changed, 'unexpected_files': extra}))
    print(package.parent)


if __name__ == '__main__':
    main()
