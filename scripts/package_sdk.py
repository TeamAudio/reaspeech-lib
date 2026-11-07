#!/usr/bin/env python3
"""Package a standalone shared C library built by the Build workflow."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'target/sdk')
    args = parser.parse_args()
    package = args.output / args.name
    if package.exists():
        parser.error(f'package already exists: {package}')
    if sys.platform == 'win32':
        runtime, location = 'reaspeech.dll', 'bin'
        libraries = ['reaspeech.dll.lib']
    elif sys.platform == 'darwin':
        runtime, location = 'libreaspeech.dylib', 'lib'
        libraries = []
    else:
        runtime, location = 'libreaspeech.so', 'lib'
        libraries = []
    files = [(args.build_dir / runtime, Path(location) / runtime)]
    files += [(args.build_dir / lib, Path('lib') / lib) for lib in libraries]
    files += [(ROOT / 'include/reaspeech.h', Path('include/reaspeech.h')),
              (ROOT / 'LICENSE', Path('LICENSE')),
              (ROOT / 'README.md', Path('README.md'))]
    example = Path('examples/c-sdk')
    for filename in ('CMakeLists.txt', 'smoke.c', 'smoke.cpp'):
        relative_path = example / filename
        files.append((ROOT / relative_path, relative_path))
    # ort's copy-dylibs feature places any shared ONNX Runtime dependencies here.
    for pattern in ('*onnxruntime*.dll', '*onnxruntime*.so*', '*onnxruntime*.dylib'):
        files += [(p, Path(location) / p.name) for p in args.build_dir.glob(pattern)]
    for source, dest in files:
        if not source.is_file():
            parser.error(f'missing build output: {source}')
    for source, dest in files:
        target = package / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    cmake = package / 'lib/cmake/ReaSpeech'
    cmake.mkdir(parents=True, exist_ok=True)
    config = '''get_filename_component(_reaspeech_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT TARGET ReaSpeech::reaspeech)
  add_library(ReaSpeech::reaspeech SHARED IMPORTED)
  set_target_properties(ReaSpeech::reaspeech PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_reaspeech_root}/include"
    IMPORTED_LOCATION "${_reaspeech_root}/RUNTIME")
'''.replace('RUNTIME', f'{location}/{runtime}')
    if libraries:
        config += '  set_target_properties(ReaSpeech::reaspeech PROPERTIES IMPORTED_IMPLIB "${_reaspeech_root}/lib/reaspeech.dll.lib")\n'
    config += 'endif()\nunset(_reaspeech_root)\n'
    (cmake / 'ReaSpeechConfig.cmake').write_text(config, encoding='utf-8')
    manifest = {str(p.relative_to(package)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(package.rglob('*')) if p.is_file()}
    (package / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    archive = Path(shutil.make_archive(str(args.output / args.name), 'zip', args.output, args.name))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')


if __name__ == '__main__':
    main()
