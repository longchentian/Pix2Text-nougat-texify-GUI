import torch
import argparse




def parse_option():
    parser = argparse.ArgumentParser(prog="nougat inference config", description="model archiver")
    parser.add_argument("--pretrained_model_name_or_path", default="./models/nougat_latex/")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--mode", default="1",help='Use which model')
    parser.add_argument('-t', '--temperature', type=float, default=0.0, help='Softmax sampling frequency')
    return parser.parse_args()

def main():
    args = parse_option()
    if torch.cuda.is_available():
        args.device = 'gpu'
    else:
        args.device = 'cpu'

    from gui_nougat import main   
    main(args)

if __name__ == '__main__':
    main()
