import cv2
import os

# Open the video file
video_file = 'video/2.mp4'
cap = cv2.VideoCapture(video_file)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a directory to store frames
if not os.path.exists('frames'):
    os.mkdir('frames')

frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if we have reached the end of the video
    if not ret:
        break

    # Save the frame as an image
    frame_filename = os.path.join('frames', f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    print(frame_count)
    frame_count += 1

# Release the video file and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(f"Total frames extracted: {frame_count}")
