
from sift import siftSolver
from sift_KeyPoints_And_Detectors import sift_KeyPoints_And_Detectors

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
      else:
          print("Unkown command")

    except:
      print("Command not found")
    selectedOption = input("Please select a question(1/2/3) or type exit to stop: ")



