import os
import re
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
import csv

from yancc.solution import DKE_OUTPUTS, MDKE_OUTPUTS


def _sizes_tex(sizes):
    out = "("
    for s in sizes:
        if s == "ns":
            out += "n_s,"
        elif s == "nx":
            out += "n_x,"
        elif s == "na":
            out += "n_{\\alpha},"
        elif s == "nt":
            out += "n_{\\theta},"
        elif s == "nz":
            out += "n_{\\zeta},"
        else:
            out += str(s) + ","
    if out[-1] == ",":
        out = out[:-1]
    return out + ")"


def _escape(line):
    match = re.findall(r"\|.*\|", line)
    if match:
        sub = r"\|" + match[0][1:-1] + "|"
        line = line.replace(match[0], sub)
    return line


def write_csv(outputs_dict, name):
    with open(name + ".csv", "w", newline="") as f:
        fieldnames = [
            "Name",
            "Label",
            "Units",
            "Description",
            "Size",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        keys = outputs_dict.keys()
        for key in keys:
            d = {
                "Name": "``" + key + "``",
                "Label": ":math:`" + outputs_dict[key]["label"].replace("$", "") + "`",
                "Units": ":math:`"
                + "\\mathrm{"
                + outputs_dict[key]["units"].replace("$", "")
                + "}"
                + "`",
                "Description": outputs_dict[key]["description"],
                "Size": ":math:`" + _sizes_tex(outputs_dict[key]["dim"]) + "`",
            }
            # stuff like |x| is interpreted as a substitution by rst, need to escape
            d["Description"] = _escape(d["Description"])
            writer.writerow(d)


header = r"""
List of Output Variables
########################

The table below contains a list of outputs from the solution of the drift kinetic
equation.

  * **Name** : Name of the variable as it appears in the code. Pass a string with this
    name to ``sol.get`` to compute the quantity.
  * **Label** : TeX label for the variable.
  * **Units** : Physical units for the variable.
  * **Description** : Description of the variable.
  * **Size** : Size of the returned array.

"""

block = """

{}
{}

.. csv-table:: List of Outputs: {}
   :file: {}.csv
   :widths: 23, 15, 15, 60, 15
   :header-rows: 1

"""

write_csv(DKE_OUTPUTS, "DKE")
header += block.format(
    "Drift Kinetic Equation (``yancc.solution.DKESolution``)",
    "-" * len("Drift Kinetic Equation (``yancc.solution.DKESolution``)"),
    "DKE",
    "DKE",
)
write_csv(MDKE_OUTPUTS, "MDKE")
header += block.format(
    "Monoenergetic Drift Kinetic Equation (``yancc.solution.MDKESolution``)",
    "-" * len("Monoenergetic Drift Kinetic Equation (``yancc.solution.MDKESolution``)"),
    "MDKE",
    "MDKE",
)

with open("variables.rst", "w+") as f:
    f.write(header)
