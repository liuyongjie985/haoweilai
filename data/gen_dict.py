import json

label_file = "./haoweilai_train_caption.txt"
fp2 = open(label_file, 'r')
labels = json.load(fp2)
fp2.close()

label_dict = {}
for x in labels:
    for y in x[1]:
        label_dict[y] = 1

label_file = "./haoweilai_test_caption.txt"
fp2 = open(label_file, 'r')
labels = json.load(fp2)
fp2.close()

for x in labels:
    for y in x[1]:
        label_dict[y] = 1

o = open("haoweilai_dict" + str(len(label_dict) + 1) + ".txt", "w")
o.write("<eol>")
o.write("\t")
o.write("0")
count = 0
o.write("\n")
for k, v in label_dict.items():
    count += 1
    o.write(k)
    o.write("\t")
    o.write(str(count))
    if count != len(label_dict):
        o.write("\n")
o.close()
