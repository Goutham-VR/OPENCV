import cv2
from ultralytics import YOLO

# --------------------------------
# CONFIGURATION
# --------------------------------

# 1 = Stop when first object is detected
# 2 = Continuous object detection
# 3 = Continuous detection + object counting

MODE = 3

# 0 = Webcam
# Or provide a video path, e.g. "video.mp4"

VIDEO_SOURCE = 0

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open camera or video
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Cannot open camera or video")
    exit()

# Store unique object tracking IDs
counted_ids = set()

print("Object detection started")

# --------------------------------
# MAIN VIDEO LOOP
# --------------------------------

while True:

    # Read one frame
    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        print("Video ended")
        break

    # --------------------------------
    # SCENARIO 1
    # --------------------------------

    if MODE == 1:

        # Detect objects in current frame
        results = model.predict(frame, verbose=False)

        # Get detection results
        result = results[0]

        # Check whether any object is detected
        if len(result.boxes) > 0:

            print("Object detected!")

            # Draw bounding boxes
            annotated_frame = result.plot()

            # Display detection frame
            cv2.imshow(
                "Object Detection",
                annotated_frame
            )

            # Stop reading new frames
            print("Detection stopped")

            # Wait until user presses Q
            while True:

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            break

        else:

            # Display normal footage
            cv2.imshow("Object Detection", frame)

    # --------------------------------
    # SCENARIO 2
    # --------------------------------

    elif MODE == 2:

        # Detect objects continuously
        results = model.predict(frame, verbose=False)

        # Draw detection boxes
        annotated_frame = results[0].plot()

        # Display live footage
        cv2.imshow(
            "Continuous Detection",
            annotated_frame
        )

    # --------------------------------
    # SCENARIO 3
    # --------------------------------

    elif MODE == 3:

        # Track objects across frames
        results = model.track(
            frame,
            persist=True,
            verbose=False
        )

        result = results[0]

        # Draw bounding boxes
        annotated_frame = result.plot()

        # Check whether tracking IDs exist
        if result.boxes.id is not None:

            # Get object tracking IDs
            object_ids = result.boxes.id.int().cpu().tolist()

            # Add new IDs to our set
            for object_id in object_ids:

                counted_ids.add(object_id)

        # Calculate unique object count
        total_objects = len(counted_ids)

        # Display object count
        cv2.putText(
            annotated_frame,
            f"Total Objects: {total_objects}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display live footage
        cv2.imshow(
            "Detection + Object Counting",
            annotated_frame
        )

    # --------------------------------
    # EXIT CONDITION
    # --------------------------------

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------
# CLEANUP
# --------------------------------

cap.release()

cv2.destroyAllWindows()

print("Total unique objects:", len(counted_ids))

print("Detection completed")