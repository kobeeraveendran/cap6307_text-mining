from pyrouge import Rouge155

import time
import glob
import json

rouge = Rouge155()

output_dicts = {}

# generates ROUGE scores for each system's summaries, directory by directory
for system in glob.glob("../System_Summaries/*/"):
    rouge.system_dir = system
    rouge.system_filename_pattern = "d(\d+)t.[A-Za-z]+"

    rouge.model_dir = "../Human_Summaries/eval/"
    rouge.model_filename_pattern = "D#ID#.M.100.T.[A-Z]"

    start = time.time()

    output = rouge.convert_and_evaluate()

    end = time.time()

    # show avg. results across each metric, time elapsed, etc.
    print("RESULTS FOR {}".format(system))
    print("TIME ELAPSED: {:.2f}s".format(end - start))
    print(output)

    # save each system's scores to a dictionary
    output_dict = rouge.output_to_dict(output)
    system_name = system.split('/')[-2]
    output_dicts[system_name] = output_dict

# save the output dictionaries as a json file and prettify it for readabillity
with open("../rouge_scores.json", "w") as output_file:
    json.dump(output_dicts, output_file, indent = 4)