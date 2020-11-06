from pyrouge import Rouge155

import glob

rouge = Rouge155()


for system in glob.glob("../System_Summaries/*"):
    rouge._system_dir = system
    rouge._model_dir = "../Human_Summaries/eval/"

    # TODO: match filename patterns, account for different expert annotations per summary document