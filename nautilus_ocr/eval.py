import glob
from collections import namedtuple, defaultdict
from dataclasses import dataclass
from typing import List

import pandas as pd
from edit_distance import edit_distance, SequenceMatcher
import numpy as np

from nautilus_ocr.word_dictionary import DictionaryCorrector


class Sync:
    def __init__(self, texts, substr=None, match=None):
        self.texts = texts

        if substr:
            assert (substr.shape[0] == len(self.texts))
            self.substr = substr
        else:
            self.substr = np.zeros((len(texts), 3), dtype=int)

        self.match = match

    def __str__(self):
        return str(self.substr)

    def get_text(self):
        return [self.texts[i][start:start + length] for i, (start, end, length) in enumerate(self.substr)]

    def is_valid(self):
        return np.any(self.substr[:, 2] > 0)

    def lengths(self):
        return self.substr[:, 2]

    def start(self, idx):
        return self.substr[idx, 0]

    def stop(self, idx):
        return self.substr[idx, 1]

    def length(self, idx):
        return self.substr[idx, 2]

    def set_start(self, idx, v):
        self.substr[idx, 0] = v

    def set_stop(self, idx, v):
        self.substr[idx, 1] = v

    def set_length(self, idx, v):
        self.substr[idx, 2] = v

    def set_all(self, idx, v):
        self.substr[idx, :] = v


def synchronize(texts):
    def init():
        sync = Sync(texts)
        for i, text in enumerate(texts):
            sync.set_all(i, [0, len(text) - 1, len(text)])

        if sync.is_valid():
            return [sync]

        return []

    def longest_match(maxlen, c1, start1, stop1, c2, start2, stop2):
        mstart1 = 0
        mstart2 = 0
        s1limit = stop1 - maxlen
        s2limit = stop2 - maxlen
        for s1 in range(start1, s1limit + 1):
            for s2 in range(start2, s2limit + 1):
                if c1[s1] == c2[s2]:
                    i1 = s1 + 1
                    i2 = s2 + 1
                    while i1 <= stop1 and i2 <= stop2 and c1[i1] == c2[i2]:
                        i1 += 1
                        i2 += 1

                    increase = i1 - s1 - maxlen
                    if increase > 0:
                        s1limit -= increase
                        s2limit -= increase
                        maxlen += increase
                        mstart1 = s1
                        mstart2 = s2

        return maxlen, mstart1, mstart2

    def save_match(synclist, num_text, sync, start, length, match):
        left, right = Sync(texts), Sync(texts)
        for i in range(num_text):
            stop = start[i] + length - 1
            left.set_all(i, [sync.start(i), start[i] - 1, start[i] - sync.start(i)])
            right.set_all(i, [stop + 1, sync.stop(i), sync.stop(i) - stop])
            sync.set_all(i, [start[i], stop, length])

        sync.match = match
        if left.is_valid():
            synclist.insert(synclist.index(sync), left)

        if right.is_valid():
            synclist.insert(synclist.index(sync) + 1, right)

    def recursive_sync(synclist, texts, start_index):
        sync = synclist[start_index]
        if np.any(sync.lengths() == 0):
            return

        start = np.zeros(len(texts), dtype=int)
        start[0] = sync.start(0)
        length = sync.length(0)
        for i, text in enumerate(texts[1:], 1):
            length, new_start, start[i] = longest_match(0, texts[0], start[0], start[0] + length - 1,
                                                        text, sync.start(i), sync.stop(i))

            if length == 0:
                return

            change = new_start - start[0]
            if change > 0:
                for j in range(i):
                    start[j] += change

        save_match(synclist, len(texts), sync, start, length, True)

        start_index = synclist.index(sync)
        if start_index - 1 >= 0:
            recursive_sync(synclist, texts, start_index - 1)

        start_index = synclist.index(sync)
        if start_index + 1 < len(synclist):
            recursive_sync(synclist, texts, start_index + 1)

        return

    synclist = init()

    if len(synclist) > 0:
        recursive_sync(synclist, texts, 0)

    return synclist


CERResult = namedtuple("CERResult", "n errs sync_errors confusion gt pred")
class WordDictEvaluator:
    @staticmethod
    def evaluate(pred: str, gt: str, skip_empty_gt: bool = False) -> CERResult:

        def sync_words(pred, gt, seperator=" "):
            p_words = pred.split(seperator)
            gt_words = gt.split(seperator)
            opt_codes = list(SequenceMatcher(p_words, gt_words).get_opcodes())
            synced_word_list = []
            for i in opt_codes:
                if i[0] == "equal" or i[0] == "replace":
                    synced_word_list.append((p_words[i[1]: i[2]], gt_words[i[3]: i[4]]))
            #print(synced_word_list)
            #print(p_words)
            #print(list(SequenceMatcher(p_words, gt_words).get_matching_blocks()))
            #print(list(SequenceMatcher(p_words, gt_words).get_opcodes()))

            #print(edit_distance(p_words, gt_words))

            return synced_word_list

        synced_word_list = sync_words(pred, gt)
        return synced_word_list

class CEREvaluator:
    @staticmethod
    def evaluate(*, pred: str, gt: str, skip_empty_gt: bool = False) -> CERResult:
        confusion = defaultdict(int)
        total_sync_errs = 0

        if len(gt) == 0 and skip_empty_gt:
            return CERResult(0, 0, 0, confusion, gt, pred)

        errs, trues = edit_distance(gt, pred)
        synclist = synchronize([gt, pred])
        for sync in synclist:
            gt_str, pred_str = sync.get_text()
            if gt_str != pred_str:
                key = (pred_str, gt_str)
                total_sync_errs += max(len(gt_str), len(pred_str))
                confusion[key] += 1

        return CERResult(len(gt), errs, total_sync_errs, confusion, gt, pred)


@dataclass
class TextLine:
    text: str
    id: str = ""


class StatTracker:
    def __init__(self, keys: tuple = ()):
        self.stats = []
        self.keys = keys

    def add(self, stat: tuple):
        self.stats.append(stat)

    def getall(self):
        return self.stats

    @staticmethod
    def merge_from(trackers):
        out = StatTracker()
        for tracker in trackers:
            if tracker is not None:
                if not out.keys:
                    out.keys = tracker.keys
                else:
                    if not out.keys == tracker.keys:
                        raise KeyError("Stat tracker keys are not compatible")
                out.stats.extend(tracker.stats)
        return out

    # prepend a column which contains all the same data
    def prepend_identifier(self, key, data):
        self.stats = [(data, *stat) for stat in self.stats]
        self.keys = (key, *self.keys)

    def get_report(self) -> pd.DataFrame:
        return pd.DataFrame(self.stats, columns=self.keys)


EvalSample = namedtuple("EvalSample", "pred gt")


class SampleCounter:
    def __init__(self, *, samples=None):
        self.samples = samples or defaultdict(int)

    def add(self, *, prediction, ground_truth, weight=1):
        self.uadd(p=prediction, gt=ground_truth, weight=weight)

    def uadd(self, p, gt, weight=1):
        self.samples[EvalSample(p, gt)] += weight

    def merge_from(self, other: 'SampleCounter'):
        for k, v in other.samples.items():
            self.samples[k] += v

    def get(self, *, prediction, ground_truth) -> int:
        return self.uget(p=prediction, gt=ground_truth)

    def uget(self, p, gt) -> int:
        return self.samples[EvalSample(p, gt)]

    # return the sum of all weights
    def sum(self, *, p, gt) -> int:
        return sum(item[1] for item in self.samples.items() if \
                   (p is None or item[0].pred == p) and \
                   (gt is None or item[0].gt == gt))

    def sum_pred_gt_equal(self) -> int:
        return sum(item[1] for item in self.samples.items() if item[0].pred == item[0].gt)

    def sum_pred_gt_notqual(self) -> int:
        return sum(item[1] for item in self.samples.items() if item[0].pred != item[0].gt)

    @staticmethod
    def merge(counters: List['SampleCounter']) -> 'SampleCounter':
        result = SampleCounter()
        for counter in counters:
            result.merge_from(counter)
        return result


EvaluationResult = namedtuple("EvaluationResult", "counter stats")


class TextLinesOCREvaluation():
    def __init__(self, skip_empty_gt: bool = False):
        super().__init__()
        self.skip_empty_gt = skip_empty_gt

    def evaluate(self, prediction: List[TextLine], gt: List[TextLine]) -> EvaluationResult:
        tracker = StatTracker(("pred", "gt", "n", "errs", "sync_errors", "id_pred", "id_gt"))
        confusion_counters = []

        for p, gt in zip(prediction, gt):
            cer_result = CEREvaluator.evaluate(pred=p.text, gt=gt.text, skip_empty_gt=self.skip_empty_gt)
            tracker.add((p.text, gt.text, cer_result.n, cer_result.errs, cer_result.sync_errors, p.id, gt.id))
            confusion_counters.append(SampleCounter(samples=cer_result.confusion))
        confusion_merged = SampleCounter.merge(confusion_counters)

        return EvaluationResult(confusion_merged, tracker)


EvaluationResultMultiModelSummary = namedtuple("EvaluationResultMultiModelSummary", "n error syncerror")

if __name__ == "__main__":

    a = "test test test"
    b = "test terst test"
    WordDictEvaluator().evaluate(a, b)

    res = []
    model = []
    calamari_suffix = ".pred.txt"
    suffix_list = ["_model1.txt", "_model2.txt", "_model3.txt", "_model4.txt", "_model5.txt",
                   "_model6.txt", "_model7.txt", "_model8.txt", "_model9.txt"]
    suffix_list_extended = []
    for x in suffix_list:
        for y in ["_dictionary_", "_unprocessed_", "_normalized_", "_segmented_"]:
            for z in ["_greedy_", "_word_beam_search_", "_beam_search_"]:
                print(x)
                suffix_list_extended.append(y+z+x)

    suffix_list_extended.append(calamari_suffix)
    print(suffix_list_extended)
    for suffix in suffix_list_extended:
        print(suffix)
        gt_lines = []
        pred_lines = []
        for pred in sorted(glob.glob("/tmp/images/*" + suffix)):
            print(pred)
            gt = pred.replace(suffix, ".gt.txt")
            print(gt)
            pred_str = ""
            gt_str = ""
            with open(pred, "r") as file:
                pred_str = file.read()
            with open(gt, "r") as file:
                gt_str = file.read()
            # print(pred_str)
            # print(gt_str)
            gt_lines.append((TextLine(gt_str)))
            pred_lines.append((TextLine(pred_str)))

            evaluator = TextLinesOCREvaluation()
            result = evaluator.evaluate(pred_lines, gt_lines)
            df = result.stats.get_report()
            df.to_csv("results" + suffix.replace("txt", "csv"), sep='\t')
            # print(df["n"])
            # results = EvaluationResultMultiModelSummary(df.sum(axis=2), df.sum(axis=3), df.sum(axis=4))
            # res.append(results)

    print(res)

    # result = CEREvaluator.evaluate(pred = pred_str, gt=gt_str)
    # print(result)
