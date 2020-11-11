from pyrouge import Rouge155

import time

import glob

rouge = Rouge155()

# rouge.model_dir = "../Human_Summaries/eval/"
# rouge.model_filename_pattern = "D(\d+).M.100.T.[A-Za-z]+"

output_dicts = []

for system in glob.glob("../System_Summaries/*/"):
    rouge.system_dir = system
    rouge.system_filename_pattern = "d(\d+)t.[A-Za-z]+"

    rouge.model_dir = "../Human_Summaries/eval/"
    rouge.model_filename_pattern = "D#ID#.M.100.T.[A-Z]"

    start = time.time()

    output = rouge.convert_and_evaluate()

    end = time.time()

    print("RESULTS FOR {}".format(system))
    print("TIME ELAPSED: {:.2f}".format(end - start))
    print(output)
    output_dict = rouge.output_to_dict(output)
    output_dicts.append(output_dict)

# print("OUTPUT_DICTS")
# print(output_dicts)