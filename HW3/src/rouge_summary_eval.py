from pyrouge import Rouge155

import glob

rouge = Rouge155()

rouge.model_dir = "../Human_Summaries/eval/"
rouge.model_filename_pattern = "D(\d+).M.100.T.(\w)"

output_dicts = []

for system in glob.glob("../System_Summaries/*/"):
    rouge.system_dir = system
    rouge.system_filename_pattern = "d(\d+)t.(\w+)"

    print("RESULTS FOR {}".format(system))
    output = rouge.convert_and_evaluate()
    print(output)
    output_dict = rouge.output_to_dict(output)
    output_dicts.append(output_dict)

print("OUTPUT_DICTS")
print(output_dicts)