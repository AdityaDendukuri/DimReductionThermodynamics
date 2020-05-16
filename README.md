# Dimensionality reduction in thermodynamically small systems 

Thermodynamically small systems are physical systems where the number of atoms gets so small (n < 100) that classical structural variables such as volume and surface area loose meaning. Since, classical theormodynamic state equations relate changes in energy with changes in volume and surface area, new variables are needed to build these relations for thermodynamically small systems. 

The idea that comes to mind is to simply use the individual coordinates of each atom as a set of structural variables to represent changes in energy. However, the dimensionality of this space is very high. This project adresses this problem using dimensionality reduction. We use two techniques. (1) Diffusion Maps; (2) Neural Networks. 

Another idea in this project is the introduction of spectral distance as a measure to capture structural similarities between two atomic clusters. This spectral distance is used to define a markov matrix over the dataset. 

The project is compiled into jupyter ipython notebooks for the various datasets used and can be accessed in /src/.

We mainly worked with 3 datasets 

  1. LJ3 : three particle lennard jones system simulated at T* = 0.18.
  2. LJ13 : three particle lennard jones system simulated at T* = 0.28.
  3. LJ13 : three particle lennard jones system simulated at T* = 0.4. 

Please email me for questions : dendukuri.aditya123@outlook.com 
