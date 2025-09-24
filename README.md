This program is made to be a program that detects the position of the arms and hands, and finds the angles between the different joints and saves them. Using the MediaPipe landmarks, the 3d points will be found and the angles between the landmarks will be saved to be used in accordance to a set of robotic hands from TMH for a Make-A-Wish demo.

    This program using MediaPipe and OpenCV to find and estimate the position of the arms
    and hands, and the data that I will use will be:
    - External shoulder rotation
    - Forward shoulder rotation
    - Elbow rotation
    - Hand state (open / closed)

Usage:

Connect a laptop and the Arduino Mega on the robotic hands using a USB A to B cable. It is also necessary for the robotic hands to have the red and black wires connected to a power supply from the breadboard, ideally running at 6 volts and 5 amps (but no higher than 6.5 volts!). 

Next, you need to run the python program (ideally from a command line) using the specified argument format: 
  
    python armdetection.py --port COM7 --baudrate 115200 

    OR running with the default baudrate of 115200

    python armdetection.py --port COM3

    OR running only on the laptop:

    python armdetection.py

This is necessary to specify the COM port the Arduino Mega is connected to, and to specify the baudrate (although the baudrate will, by default be 115200 so it does not necessarily need specified.) The program will then display a window with the arm detection and communicate to the Arduino, running the demonstration.



DEPENDENCIES: 

    pip install opencv-python
    pip install mediapipe
    pip install numpy
    pip install serial
