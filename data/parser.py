import numpy as np

file = open("T029.txt", "r")
file_w = open("13P_T029_POTENTIALS.txt", "w")

lines = file.readlines()

features = []
lables = []

for iter, x in enumerate(lines):
      data = x.split()
      print(data)
      if iter%13 == 0:
            file_w.write(data[3] + "\n")


