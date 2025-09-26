import cv2

def change_video_speed(input_path, output_path, target_fps=30, target_duration=1.0):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_duration = frame_count / original_fps

    speed_factor = original_duration / target_duration

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Calculate new frame count: 30 frames for 1 second
    new_frame_count = int(target_fps * target_duration)

    for i in range(new_frame_count):
        # Map output frame index to original frame index
        original_frame_index = int(i * speed_factor)
        if original_frame_index >= len(frames):
            original_frame_index = len(frames) - 1
        out.write(frames[original_frame_index])

    out.release()
    print(f"Video saved to {output_path}")

# Example usage
change_video_speed('input.mp4', 'output_1sec.mp4')
