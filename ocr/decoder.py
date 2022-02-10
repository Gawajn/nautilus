from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import groupby
from typing import List

import numpy as np
from word_beam_search import WordBeamSearch

from modules.CTCDecoder.ctc_decoder.beam_search import beam_search
from modules.CTCDecoder.ctc_decoder.bk_tree import BKTree
from modules.CTCDecoder.ctc_decoder.language_model import LanguageModel


@dataclass
class CharLocationInfo:
    char_start: int
    char_end: int
    char: str


@dataclass
class ExtendedDecoderInfo:
    chars: str
    white_space_index: int
    blank_index: int
    charLocationInfo: List[CharLocationInfo]


@dataclass
class DecoderOutput:
    decoded_string: str
    char_mapping: ExtendedDecoderInfo = None


class Decoder:
    def __init__(self, chars):
        self.chars = chars

    @abstractmethod
    def decode(self, mat):
        pass


class GreedyDecoder(Decoder):
    def __init__(self, chars):
        super().__init__(chars)

    def decode(self, mat, extendend_info=False):
        index_list = np.argmax(np.squeeze(mat), axis=1)
        blank_index = len(self.chars)
        best_chars_collapsed = [self.chars[k] for k, _ in groupby(index_list) if k != blank_index]
        pred_string = ''.join(best_chars_collapsed)

        extendend_info_stats = None
        if extendend_info:
            listextf = []

            start, end = 0, -1
            whitespace_char_index = self.chars.find(" ")
            prev = None
            for x, y in groupby(index_list):
                y_length = len(list(y))
                prev = x if prev is None else prev
                start = end + 1
                end = end + y_length
                listextf.append(CharLocationInfo(start, end, x))

            listextf = [x for x in listextf if x.char != blank_index]
            listextf = [x for x in listextf if x.char != whitespace_char_index]

            extendend_info_stats = ExtendedDecoderInfo(chars=self.chars,
                                                       white_space_index=whitespace_char_index,
                                                       blank_index=blank_index,
                                                       charLocationInfo=listextf)

        print(f'Best path: "{pred_string}"')
        return DecoderOutput(decoded_string=pred_string, char_mapping=extendend_info_stats)


class BeamSearchDecoder(Decoder):
    def __init__(self, chars, beam_width, corpus):
        super().__init__(chars)
        self.beam_width = beam_width
        self.corpus = corpus
        self.lm = LanguageModel(self.corpus, self.chars)

    def decode(self, mat):
        pred_string = beam_search(np.squeeze(mat), self.chars, beam_width=self.beam_width, lm=self.lm)
        print(f'Beam search: {pred_string}')
        return DecoderOutput(pred_string)


class WordBeamSearchDecoder(Decoder, ABC):
    def __init__(self, chars, beam_width, corpus, non_word_chars=" "):
        super().__init__(chars)
        self.beam_width = beam_width
        self.corpus = corpus
        self.non_word_chars = non_word_chars
        self.word_chars = ''.join(set(self.chars).difference(self.non_word_chars))

        self.wbs = WordBeamSearch(beam_width, 'Words', 0.0, self.corpus.encode('utf8'), self.chars.encode('utf8'),
                                  self.word_chars.encode('utf8'))

    def decode(self, mat):
        char_str = []
        label_str = self.wbs.compute(mat)
        for curr_label_str in label_str:
            s = ''.join([self.chars[label] for label in curr_label_str])
            char_str.append(s)
        return DecoderOutput(''.join(char_str))


class LexiconDecoder(Decoder, ABC):
    def __init__(self, chars, corpus, tolerance=2):
        super().__init__(chars)
        self.corpus = corpus
        self.tolerance = tolerance
        self.bk_tree = BKTree(list(set(self.corpus.split(" "))))
        self.grd_decoder = GreedyDecoder(self.chars)

    def decode(self, mat: np.ndarray) -> str:
        """Adaption of Lexicon search decoder.

        The algorithm computes a first approximation using best path decoding. Similar words are queried using the BK tree.
        These word candidates are then scored given the neural network output, and the best one is returned.
        See CRNN paper from Shi, Bai and Yao.

        Args:
            mat: Output of neural network of shape TxC.
            chars: The set of characters the neural network can recognize, excluding the CTC-blank.
            bk_tree: Instance of BKTree which is used to query similar words.
            tolerance: Words to be considered, which are within specified edit distance.

        Returns:
            The decoded text.
        """
        # todo
        '''
        # use best path decoding to get an approximation
        decoder_output = self.grd_decoder.decode(mat)
        # print(best_path_indices)

        t_word_length = 0
        for i in decoder_output.decoded_string.split(" "):
            word_length = len(i)
            t_word_length += word_length
            best_chars_collapsed = [self.chars[k.char] for k in decoder_output.char_mapping.charLocationInfo[t_word_length - word_length:t_word_length]]
            span =  decoder_output.char_mapping.charLocationInfo[t_word_length - word_length:t_word_length]
            mini, maxi = min(span, key=lambda t: t.start), max(span, key=lambda t: t.end)
            words = self.bk_tree.query(i, self.tolerance)
            print(words)

            ##print(mat.shape)
            # print(mat[mini.start:maxi.end,:].shape)
            def consistsWordofonlyChars(word, chars):
                flag = True
                for w in word:
                    if w not in chars:
                        flag = False
                        break
                return flag

            words = [w for w in words if consistsWordofonlyChars(w, chars)]
            words.append(i)
            print(words)
            word_probs = []
            for w in words:
                t = approx.replace(i, w)
                print(listextf)
                print(word_length)
                print(t_word_length)
                print(w, loss(mat, t, chars))
            print("original", loss(mat, approx, chars))

            for w in words:
                print(listextf)
                print(word_length)
                print(t_word_length)
                print(w)
                print(w, loss(mat[mini.start:maxi.end, :], w, chars))
                (w, loss(mat[mini.start:maxi.end, :], w, chars))
            word_probs = [(w, loss(mat[mini.start:maxi.end,:], w, chars)) for w in words]
            word_probs.sort(key=lambda x: x[1], reverse=True)
            ##word_probs2 = [(w, probability2(mat[mini.start:maxi.end,:], w, chars)) for w in words]
            #word_probs2.sort(key=lambda x: x[1], reverse=True)
            print("123")
            print(word_probs)

        # if there are no similar words, return empty string
        if not words:
            return ''

        # else compute probabilities of all similar words and return best scoring one
        word_probs = [(w, probability(mat, w, chars)) for w in words]
        word_probs.sort(key=lambda x: x[1], reverse=True)
        '''
        return None
