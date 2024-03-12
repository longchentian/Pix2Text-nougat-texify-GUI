import subprocess
import os

import streamlit.web.cli as stcli
import os, sys
import texify
# def run_app():
#     cur_dir = os.path.dirname(os.path.abspath(__file__))
#     ocr_app_path = os.path.join(cur_dir, "ocr_app.py")
#     subprocess.run(["streamlit", "run", ocr_app_path],shell=True)
# import ocr_app

def resolve_path(path):
    # resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_path = os.path.join(cur_dir, "ocr_app.py")
    return resolved_path

if __name__ == '__main__':
    sys.argv = [
            "streamlit",
            "run",
            resolve_path("ocr_app.py"),
            "--global.developmentMode=false",
        ]
    sys.exit(stcli.main())

