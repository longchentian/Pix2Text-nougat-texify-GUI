# -*- mode: python ; coding: utf-8 -*-


from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata
 
datas = [("C:/Anaconda3/envs/trial/Lib/site-packages/streamlit/runtime","./streamlit/runtime")]
datas += collect_data_files("streamlit")
datas += copy_metadata("streamlit")
#("C:/Anaconda3/envs/trial/Lib/site-packages/streamlit/static","./streamlit/static")

a = Analysis(
    ['main.py'],
    pathex=['texify_pdf_ocr_app.py', 'gui.py','texify_app.py','nougat_app.py','pix2text_app.py','ocr_app.py','gui_nougat.py','gui_texify.py','gui_pix2text.py'],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=['./hooks/'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
