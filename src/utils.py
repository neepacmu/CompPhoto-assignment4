import cv2
import os

video_path = '/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn4/video/DSC_0530.MOV'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    exit()

output_directory = 'output_frames_2'
os.makedirs(output_directory, exist_ok=True)

frame_count = -1
while True:
    

    ret, frame = cap.read()

    #cv2.resize(frame, ())
    #
    if frame is not None:
        frame = frame[::1,::1]
    #print(frame.shape)
    if not ret:
        break
    frame_count += 1
    if frame_count % 1 == 0:
        pass
    else:
        continue
    frame_filename = os.path.join(output_directory, f'frame_{frame_count:04d}.jpg')

    cv2.imwrite(frame_filename, frame)

    

    

print(frame_count)
cap.release()
cv2.destroyAllWindows()