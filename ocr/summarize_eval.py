import glob

if __name__ == "__main__":

    for csv in sorted(glob.glob("*csv")):
        with open(csv, "r") as file:
            line = file.readline()
            line = file.readline()

            t_num = 0
            t_err = 0
            t_sync_err = 0
            while line != '':
                num, errs, syncerrors = line.split("\t")[-5:-2]
                t_num += int(num)
                t_err += int(errs)
                t_sync_err += int(syncerrors)
                line = file.readline()

        print("Modelid: {} Total Chars: {} Total Errs {} Total Sync Errors {} Char Accuracy {} Sync Accuracy {} ".format(csv, t_num, t_err, t_sync_err, 1 - t_err / t_num, 1 - t_sync_err / t_num))
