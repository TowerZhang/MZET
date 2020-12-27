import numpy as np


def label_frequence(file, new_file, only_statis=False):
    lines = []
    label_list = []
    ratio = []
    fr = open(file, 'r')
    for line in fr:
        # '8286', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1', 'chairman', 'Bruno DeGol , DeGol Brothers Lumber , Gallitzin'
        seg = line.replace("\n", "").split("\t")[1:]
        # '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1', 'chairman', 'Bruno DeGol , DeGol Brothers Lumber , Gallitzin'
        lines.append(seg)
        label_list.append([int(x) for x in seg[0].split(",")])
    label_array = np.array(label_list)
    label_sum = np.sum(label_array, axis=0)
    most_frequency = max(label_sum)
    print(label_sum, most_frequency)
    # ratio = most_frequency/label_sum
    ratio = label_sum  / most_frequency
    print(ratio)
    for i in range(len(ratio)):
        if ratio[i] >= 10e4:
            ratio[i] = 10e4
        elif ratio[i] < 10000 and ratio[i] >= 1000:
            ratio[i] = 1000
        elif ratio[i] < 1000 and ratio[i] >= 100:
            ratio[i] = 100
        elif ratio[i] < 100 and ratio[i] >= 10:
            ratio[i] = 10
        elif ratio[i] < 10 and ratio[i] >= 1:
            ratio[i] = 1
    fr.close()

    if only_statis:
        fw = open(new_file, 'w')
        count = 0
        for line in lines:
            label = [int(x) for x in line[0].split(",")]
            idx = np.where(np.array(label) == 1)
            repeat = int(max(ratio[idx]))
            middle = "\t".join([x for x in line])
            for i in range(repeat):
                fw.write(str(count) + "\t" + middle + "\n")
                count += 1
        fw.close()


if __name__ == "__main__":
    file = "Data/BBN/intermediate/test.tsv"
    new_file = "Data/BBN/intermediate/train_sample.tsv"
    label_frequence(file, new_file, only_statis=False)
