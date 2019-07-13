import numpy as np

file = open("Lj_018f.txt", "r")
file_w = open("lables.txt", "w")

lines = file.readlines()

features = []
lables = []

for iter, x in enumerate(lines):
      data = x.split()
      print(data)
      if iter%3 == 0:
            file_w.write(data[3] + "\n")


