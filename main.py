import torch
import argparse
import os, sys
import streamlit.web.cli as stcli

def parse_option():
    parser = argparse.ArgumentParser(prog="nougat inference config", description="model archiver")
    parser.add_argument("--pretrained_model_name_or_path", default="./models/nougat_latex/")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--mode", default="1",help='Use which model')
    parser.add_argument('-t', '--temperature', type=float, default=0.0, help='Softmax sampling frequency')
    return parser.parse_args()

def app():
    args = parse_option()
    if torch.cuda.is_available():
        args.device = 'gpu'
    else:
        args.device = 'cpu'
    print("Please select the model you want to use!")
    print("1:Nougat-Latex-OCR-|-Supports printed formulas")
    print("2:Texify-|-Supports printed formulas")
    print("3:Pix2Text-mfd-yolo7-tiny-mfr|Supports printed and handwritten formulas")
    print("4:Texify-PDF-ocr-app-|-Using web page screenshots")
    mode = int(input()) 
    args.mode = mode
    return args
if __name__ == '__main__':
    args = app()
    if args.mode==4:
        from texify_pdf_ocr_app import resolve_path
        sys.argv = [
            "streamlit",
            "run",
            resolve_path("ocr_app.py"),
            "--global.developmentMode=false",
        ]
        sys.exit(stcli.main())

    else:
        from gui import main   
        main(args)
