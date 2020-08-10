# Mouse-control-using-hand-gestures
A program to control your mouse using just your hand gestures using opencv and python



Firstly in this process , I have used background subtraction technique for recognising the hand and then convex hull to detect hand movements.
I have used pyautogui for the operations performed. So the steps you need to follow to run the program are:
1. Clone the repository and using cmd go into the directory.
2. Execute python hand_gesture.py
3. Set the position accordingly considering the blue box as main area of interest .
4. Make sure you don't have any moving objects in the background and good lighting conditions.
5. Press 's' to capture the background and 'r' to reset the background.
The movements you can perform are as follows:
- Two fingers - Move the mouse on the screen
- Three fingers - Scroll up and down on the screen
- Four fingers - Click on the screen where mouse is present
- Five fingers - Right click on the screen where mouse is present


Also you can vary the mouse sensitivity and the speed of scrolling the mouse using the trackbar given in the window.
