
from sift import siftSolver
from Q3Part1 import *
from Q3Part2 import ransacWarper
from sift_KeyPoints_And_Detectors import sift_KeyPoints_And_Detectors
import sys
selectedOption = input("Please select a question(1/2/3) or type exit to stop: ")

while selectedOption.lower() != 'exit':
    try:
      selectedOption = int(selectedOption)
      if int(selectedOption) == 2:
          imPath = input("Enter path for image to visualize intrest points: ")
          visualizer = sift_KeyPoints_And_Detectors(imPath)
          visualizer.drawKeyPoints()
          
          imPath1 = input("Enter path for image1 to start matching: ")
          imPath2 = input("Enter path for image2: ")
          matcher = input("Select test (Rati-Test = 0  bidirectional-test = 1):")
          matcher = siftSolver(imPath1, imPath2)
          matcher.matching()
      elif int(selectedOption) == 3:
          print("Solving Q3 part1 - Image warp to squares:")
          solveQ3Part1()
          imPath1 = input("Enter path for image1 to start RANSAC matching: ")
          imPath2 = input("Enter path for image2: ")
          warper = ransacWarper(imPath1, imPath2)
          warper.getBestTransform()
      else:
          print("Unkown command")

    except:
      print("UnExpected Error: ",sys.exc_info()[0])
    selectedOption = input("Please select a question(1/2/3) or type exit to stop: ")



