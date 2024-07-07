# A quantum-inspired algorithm for galaxy classification

This project is created for my bachelor thesis "A quantum-inspired algorithm for galaxy classification" at University Leiden, Netherlands. The bachelor thesis was supervised by Dr. E.P.L. van Nieuwenburg.
This project contains a classification model, the possibility to calculate the classification benchmark and methods to calculate the necessary data used in this project, all written in Python. All the tools require Python 3.10.2. 

# Setting up the project
For this python program to work you need to have downloaded the images from the following link: https://zenodo.org/records/3565489#.Y3vFKS-l0eY 
These images are then supposed to be put in a folder named images in the main folder of the project.
Along this 'images' folder, the main folder should contain the gz2_filename_mapping.csv file which is also downloaded with the above link, but it should also contain the gz2_hart16.csv file, which can be downloaded with the following link: https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz 

With these files in the correct position you can start the program.

# Using the project
When starting up the project using the command `py .\Classic_Classifier.py` ,a selection menu appears in the terminal. 
From there you can operate the the program, but the selection menu is in order of what you should perform first.
  - option 1 -> creates a seperate csv file containing only the relevant information to this project, this csv file is then used in all other functions within the program.
  - option 2 -> seperates the images, now grayscaled, into their respective class
  - option 3 -> calculates the classification accuracy baseline, which is used in the project
  - option 4 -> calculates the mean intensity of the image dataset
  - option 5 -> calculates the mean variance of the image dataset
  - option 6 -> opens a menu from which you can choose the desired compression to be run.
  - option 7 -> opens the menu to choose which uncompressed or compressed dataset you want to run through the classification algorithm.

For ease the compressed database that can be made with option 1 is already present in the repository so the large csv file gz2_hart.csv does not necessarily have to be downloaded.
