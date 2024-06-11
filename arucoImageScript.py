import cv2
import numpy as np

def detect_aruco_markers(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not open image.")
        return
    
    # Define the ArUco dictionary and the detector parameters
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters_create()
    
    # Optionally, adjust some parameters
    # parameters.adaptiveThreshWinSizeMin = 3
    # parameters.adaptiveThreshWinSizeMax = 23
    # parameters.adaptiveThreshWinSizeStep = 10
    # parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers in the image
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Print the number of detected markers for diagnostics
    print(f"Detected {len(corners)} markers")
    
    # Draw detected markers and display IDs
    if ids is not None:
        image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for i in range(len(ids)):
            c = corners[i][0]
            # Calculate the center of the marker
            center = np.mean(c, axis=0).astype(int)
            cv2.putText(image, f"ID: {ids[i][0]}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("No markers found.")
    
    # Save the image with detected markers
    output_image_path = 'detected_markers.jpg'
    cv2.imwrite(output_image_path, image)
    print(f"Saved output image to {output_image_path}")
    
    # Display the image with detected markers
    cv2.imshow('Aruco Marker Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Path to your image file
    image_path = 'video/test_image.png'
    
    # Call the function
    detect_aruco_markers(image_path)
