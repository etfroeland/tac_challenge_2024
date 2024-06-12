import cv2
import numpy as np
import mss
import mss.tools

def detect_aruco_markers_from_screen(output_file_path):
    # Define the ArUco dictionary and the detector parameters
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters_create()
    
    # Optionally, adjust some parameters
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    frame_count = 0
    detected_ids = set()

    with open(output_file_path, 'w') as file:
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # You can change this to capture a specific monitor
            while True:
                # Capture the screen
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                
                # Convert the frame to BGR format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                frame_count += 1
                
                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect ArUco markers in the frame
                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                
                if ids is not None:
                    for id in ids.flatten():
                        if id not in detected_ids:
                            # Log the first occurrence of the ID
                            file.write(f"Frame {frame_count}: Detected ID: {id}\n")
                            detected_ids.add(id)
                    
                    # Draw detected markers and display IDs
                    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    for i in range(len(ids)):
                        c = corners[i][0]
                        # Calculate the center of the marker
                        center = np.mean(c, axis=0).astype(int)
                        cv2.putText(frame, f"ID: {ids[i][0]}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display the frame with detected markers
                cv2.imshow('Aruco Marker Detection', frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Path to your output file
    output_file_path = 'output/detected_markers.txt'
    
    # Call the function
    detect_aruco_markers_from_screen(output_file_path)
