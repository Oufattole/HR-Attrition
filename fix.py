import re
myfile = "Predicting_employee_attrition.ipynb"
with open(myfile, "r") as f:
    data = f.read()

with open(myfile, "w") as f:
    f.write(data.replace(r"\r", r""))