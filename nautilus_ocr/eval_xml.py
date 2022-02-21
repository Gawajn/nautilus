import glob
from collections import defaultdict
from random import shuffle

from nautilus_ocr.eval import TextLinesOCREvaluation, TextLine, WordDictEvaluator
from nautilus_ocr.word_dictionary import DictionaryCorrector

if __name__ == "__main__":
    dc = DictionaryCorrector()
    dc.load_dict("../data/default_dictionary.json", "../data/bigram_default_dictionary.txt")
    res = []
    model = []
    suffix_list = ["_model1.txt", "_model2.txt", "_model3.txt", "_model4.txt", "_model5.txt",
                   "_model6.txt", "_model7.txt", "_model8.txt", "_model9.txt"]
    suffix_list = ["_model9.txt"]
    suffix_list_extended = []
    for x in suffix_list:
        for y in ["_unprocessed_", "_normalized_", "_segmented_", "_dictionary_"]:
            for z in ["_greedy_", "_word_beam_search_", "_beam_search_"]:
                suffix_list_extended.append(y+z+x)
    import xlsxwriter
    from xlsxwriter.utility import xl_rowcol_to_cell

    print(suffix_list_extended)
    for suffix in suffix_list:

        # Create an new Excel file and add a worksheet.
        workbook = xlsxwriter.Workbook(f'results{suffix}.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.set_column('A:A', 30)

        worksheet.set_column('B:B', 100)
        index :int  = 0
        d = defaultdict(list)
        list_of_files = glob.glob("/tmp/images/*.png")
        shuffle(list_of_files)
        for pred in list_of_files:
            worksheet.insert_image(index, 0, pred)
            index = index + 4
            worksheet.write(index, 2, "number of chars")
            worksheet.write(index, 3, "Char errors")
            worksheet.write(index, 4, "Sync errors")
            gt = pred.replace(".png", ".gt.txt")
            with open(gt, "r") as file:
                gt_str = file.read()
                worksheet.write(index, 0, 'GT')
                worksheet.write(index, 1, gt_str)
                words = gt_str.split(" ")
                not_in_lexicon= []
                for x in words:
                    if x not in dc.word_list:
                        not_in_lexicon.append(x)
                worksheet.write(index, 5, f'NotInLexicon: {" ".join(not_in_lexicon)}')

                index = index + 1

            for x in ["_greedy_", "_word_beam_search_", "_beam_search_"]:
                for y in ["_unprocessed_", "_normalized_", "_segmented_", "_dictionary_"]:
                    unprocessed_path = pred.replace(".png",y + x + suffix)
                    with open(unprocessed_path, "r") as file:
                        pred_string = file.read()
                        worksheet.write(index, 0, f'{x.replace("_", "") + "_"+ y.replace("_", "")}:')
                        worksheet.write(index, 1, pred_string)

                        evaluator = TextLinesOCREvaluation()
                        w_evaluator = WordDictEvaluator()
                        words = pred_string.split(" ")

                        not_in_lexicon = []
                        for hh in words:
                            if hh not in dc.word_list:
                                not_in_lexicon.append(hh)
                        result = evaluator.evaluate([TextLine(pred_string)], [TextLine(gt_str)])
                        df = result.stats.get_report()
                        df = result.stats.getall()
                        d[x+y].append(index)
                        worksheet.write(index, 2, df[0][2])
                        worksheet.write(index, 3, df[0][3])
                        worksheet.write(index, 4, df[0][4])
                        worksheet.write(index, 5, f'NotInLexicon: {" ".join(not_in_lexicon)}')

                        index = index + 1
            index = index + 2
        worksheet.write(0, 9, "Type")
        worksheet.write(0, 10, "Chars")
        worksheet.write(0, 11, "CharsErr")
        worksheet.write(0, 12, "SyncErr")
        worksheet.write(0, 13, "ACC")
        worksheet.write(0, 14, "SynACC")

        index = 1
        for t in d.keys():
            val = d[t]
            worksheet.write(index, 9, t)
            worksheet.write_formula(index, 10,  f"""=SUM({" ,".join([xl_rowcol_to_cell(x, 2) for x in val])})""")
            worksheet.write_formula(index, 11, f"""=SUM({" ,".join([xl_rowcol_to_cell(x, 3) for x in val])})""")
            worksheet.write_formula(index, 12, f"""=SUM({" ,".join([xl_rowcol_to_cell(x, 4) for x in val])})""")
            worksheet.write_formula(index, 13, f"""=1 -({xl_rowcol_to_cell(index, 11)} / {xl_rowcol_to_cell(index, 10)})""")
            worksheet.write_formula(index, 14, f"""=1 - ({xl_rowcol_to_cell(index, 12)} / {xl_rowcol_to_cell(index, 10)})""")
            index += 1

        workbook.close()


