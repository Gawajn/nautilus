import glob
import os
from abc import abstractmethod, ABC
from collections import namedtuple
from dataclasses import dataclass
from itertools import groupby
from typing import Any, List

from ctc_decoder import LanguageModel, beam_search, best_path, BKTree, lexicon_search

import pandas as pd
import torch
import torch.utils.data
import yaml
import numpy as np
from word_beam_search import WordBeamSearch

from ocr.decoder import WordBeamSearchDecoder, GreedyDecoder, BeamSearchDecoder
from ocr.word_dictionary import CharReplacementEngine, DictionaryCorrector
from utils import CTCLabelConverter, AttnLabelConverter, Averager, AttrDict
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        # table.add_row([name, param])
        total_params += param
        print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params




class Network:
    def __init__(self, opt, model_path, chars, corpus=''):
        """ model configuration """
        opt.select_data = opt.select_data.split('-')
        opt.batch_ratio = opt.batch_ratio.split('-')
        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
        model = torch.nn.DataParallel(model).to(device)
        pretrained_dict = torch.load(model_path)
        model.load_state_dict(pretrained_dict)
        self.model = model.to(device)
        self.batch_max_length = opt.batch_max_length
        self.model.eval()
        self.chars = chars
        self.corpus = corpus
        self.beam_width = 25

    def predict_single_image(self, img_path):
        from PIL import Image
        import torchvision.transforms as transforms
        # length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        self.toTensor = transforms.ToTensor()
        image = Image.open(img_path).convert('L')
        img = self.toTensor(image)
        img.sub_(0.5).div_(0.5)
        img = img.unsqueeze(0)
        batch_size = img.size(0)
        text_for_pred = torch.LongTensor(batch_size, self.batch_max_length + 1).fill_(0).to(device)
        output = self.model(img, text_for_pred)
        output = torch.softmax(output, dim=-1)
        output = output.permute(1, 0, 2)
        output = output.detach().cpu().numpy()
        d = np.dstack((output[:, :, 1:], output[:, :, 0]))
        grd = GreedyDecoder(self.chars)
        bsd = BeamSearchDecoder(self.chars, self.beam_width, self.corpus)
        wbsd = WordBeamSearchDecoder(self.chars, self.beam_width, self.corpus)
        grd_pred = grd.decode(d)
        bsd_pred = bsd.decode(d)
        wbsd = wbsd.decode(d)
        print(grd_pred.decoded_string)
        print(bsd_pred.decoded_string)
        print(wbsd.decoded_string)
        return grd_pred


def get_config(file_path, char=None):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if char == None:
        if opt.lang_char == 'None':
            characters = ''
            for data in opt['select_data'].split('-'):
                csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
                df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'],
                                 keep_default_na=False)
                all_char = ''.join(df['words'])
                characters += ''.join(set(all_char))
            characters = sorted(set(characters))
            opt.character = ''.join(characters)
        else:
            opt.character = opt.number + opt.symbol + opt.lang_char
    else:
        opt.character = char
    # os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


@dataclass
class Modelenum:
    m_path: str
    c_path: str
    ident_suffix: str
    char: str = None
    network = None

    def load_network(self):
        path = os.path.join('text_language_model.json')
        text = ''
        with open(path, 'r') as file:
            text = file.read().replace('\n', '')
            text = ' '.join(text.split())
        opt = get_config(self.c_path, self.char)
        if self.char == None:
            self.char = opt['character']
        self.network = Network(opt, self.m_path, self.char, corpus=text)

    def predict(self, image):
        if self.network is None:
            self.load_network()
        pred = self.network.predict_single_image(image)
        return pred


if __name__ == "__main__":
    models = []
    models.append(Modelenum(m_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/first/best_accuracy.pth",
                            c_path="/home/alexanderh/projects/EasyOCR/trainer/config_files/en_filtered_config.yaml",
                            ident_suffix="_model1"))
    models.append(Modelenum(m_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni/best_accuracy.pth",
                            c_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni/en_filtered_config.yaml",
                            ident_suffix="_model2",
                            char=' "#,-./03456789;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz§¶ÄÊßâãäæêëîóôöúûüýāēīōœūŵſƷʒ̂̃̄̈̾͛ͣͤͥͦͧͪͬͭͮͯ͞ωᷓᷠᷤᷦḡṙṽẃẏị․⁊℈ↄꝐꝑꝓꝙꝛꝝꝪꝫꝭꝰ'))
    models.append(Modelenum(m_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni2/best_accuracy.pth",
                            c_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni/en_filtered_config.yaml",
                            ident_suffix="_model3",
                            char=' "#,-./03456789;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz§¶ÄÊßâãäæêëîóôöúûüýāēīōœūŵſƷʒ̂̃̄̈̾͛ͣͤͥͦͧͪͬͭͮͯ͞ωᷓᷠᷤᷦḡṙṽẃẏị․⁊℈ↄꝐꝑꝓꝙꝛꝝꝪꝫꝭꝰ'))
    models.append(Modelenum(m_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni3/best_accuracy.pth",
                            c_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni/en_filtered_config.yaml",
                            ident_suffix="_model4",
                            char=' "#,-./03456789;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz§¶ÄÊßâãäæêëîóôöúûüýāēīōœūŵſƷʒ̂̃̄̈̾͛ͣͤͥͦͧͪͬͭͮͯ͞ωᷓᷠᷤᷦḡṙṽẃẏị․⁊℈ↄꝐꝑꝓꝙꝛꝝꝪꝫꝭꝰ'))

    models.append(Modelenum(m_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni4/best_accuracy.pth",
                            c_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni/en_filtered_config.yaml",
                            ident_suffix="_model5",
                            char=' "#,-./03456789;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz§¶ÄÊßâãäæêëîóôöúûüýāēīōœūŵſƷʒ̂̃̄̈̾͛ͣͤͥͦͧͪͬͭͮͯ͞ωᷓᷠᷤᷦḡṙṽẃẏị․⁊℈ↄꝐꝑꝓꝙꝛꝝꝪꝫꝭꝰ'))
    models.append(
        Modelenum(m_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/en_filtered/best_accuracy.pth",
                  c_path="/home/alexanderh/projects/EasyOCR/trainer/config_files/en_filtered_config_post.yaml",
                  ident_suffix="_model6",
                  char=' "#,.abcdefghiklmnopqrstuvxyzſω'))
    models.append(Modelenum(m_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni5/best_accuracy.pth",
                            c_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni/en_filtered_config.yaml",
                            ident_suffix="_model7",
                            char=' "#,-./03456789;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz§¶ÄÊßâãäæêëîóôöúûüýāēīōœūŵſƷʒ̂̃̄̈̾͛ͣͤͥͦͧͪͬͭͮͯ͞ωᷓᷠᷤᷦḡṙṽẃẏị․⁊℈ↄꝐꝑꝓꝙꝛꝝꝪꝫꝭꝰ'))
    models.append(Modelenum(m_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni6/best_accuracy.pth",
                            c_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni/en_filtered_config.yaml",
                            ident_suffix="_model8",
                            char=' "#,-./03456789;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz§¶ÄÊßâãäæêëîóôöúûüýāēīōœūŵſƷʒ̂̃̄̈̾͛ͣͤͥͦͧͪͬͭͮͯ͞ωᷓᷠᷤᷦḡṙṽẃẏị․⁊℈ↄꝐꝑꝓꝙꝛꝝꝪꝫꝭꝰ'))
    models.append(Modelenum(m_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni7_nn/best_accuracy.pth",
                            c_path="/home/alexanderh/projects/EasyOCR/trainer/saved_models/uni/en_filtered_config.yaml",
                            ident_suffix="_model9",
                            char=' "#,.abcdefghiklmnopqrstuvxyzſω'))

    models = models[-1:]
    dc = DictionaryCorrector()
    dc.load_dict("abc")

    for image in glob.glob("/tmp/images/*png"):
        for model in models:
            print(image)
            print(model.ident_suffix)
            pred = model.predict(image)
            pred2 = dc.segmentate_correct_and_hyphenate_text(pred[0].decoded_string)
            print([pred2])
            with open(image.replace(".png", model.ident_suffix + ".txt"), "w") as file:
                file.write(pred[0].decoded_string)
    pass
