# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = [
    'matplotlib',
    'matplotlib.backends',
    'matplotlib.backends.backend_pdf',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.pyplot',
] + collect_submodules('pydicom.pixel_data_handlers')

excluded_optional_packages = [
    'av',
    'cupy',
    'dask',
    'fsspec',
    'jax',
    'networkx',
    'onnxruntime',
    'pandas',
    'sklearn',
    'sympy',
    'tensorflow',
    'torch',
    'torchaudio',
    'torchvision',
    'triton',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={'matplotlib': {'backends': ['Agg']}},
    runtime_hooks=[],
    excludes=excluded_optional_packages,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FSS_Measurement',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
