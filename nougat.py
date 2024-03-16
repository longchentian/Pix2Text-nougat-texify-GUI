# -*- coding:utf-8 -*-
# create: @time: 10/8/23 11:47
import argparse

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from nougat_latex.util import process_raw_latex_code
from nougat_latex import NougatLaTexProcessor

def parse_option():
    parser = argparse.ArgumentParser(prog="nougat inference config", description="model archiver")
    parser.add_argument("--pretrained_model_name_or_path", default="./models/nougat_latex/")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--mode", default="1",help='Use which model')
    parser.add_argument('-t', '--temperature', type=float, default=.333, help='Softmax sampling frequency')
    return parser.parse_args()

class nougat_latex():
    def __init__(self, args = None):
        
        self.args = args

        if self.args.device == "gpu":
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # init model
        self.model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model_name_or_path).to(self.device)
        # init processor
        self.tokenizer = NougatTokenizerFast.from_pretrained(args.pretrained_model_name_or_path)
        self.latex_processor = NougatLaTexProcessor.from_pretrained(args.pretrained_model_name_or_path)

    def predict(self,image):
        if not image.mode == "RGB":
            image = image.convert('RGB')
        pixel_values = self.latex_processor(image, return_tensors="pt").pixel_values
        task_prompt = self.tokenizer.bos_token
        decoder_input_ids = self.tokenizer(task_prompt, add_special_tokens=False,
                                  return_tensors="pt").input_ids
        
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_length,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        sequence = self.tokenizer.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "").replace(self.tokenizer.bos_token,                                                                                        "")
        sequence = process_raw_latex_code(sequence)
        return sequence

  
if __name__ == '__main__':
    # run_nougat_latex()
    args = parse_option()
    img = Image.open("test.png") # Your image name here
    model = nougat_latex(args)
    results = model.predict(img)
    print(results)
