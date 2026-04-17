import cv2
import numpy as np

# Function to check if the object is inside the defined region
def is_in_region(x1, y1, x2, y2, region):
    """
    Check if the center of the bounding box is inside the region.
    :param x1, y1, x2, y2: Bounding box coordinates.
    :param region: Tuple with two points representing the region [(x1, y1), (x2, y2)].
    :return: True if the center of the bounding box is within the region.
    """
    # Calculate the center of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Check if the center is inside the region
    return region[0][0] <= center_x <= region[1][0] and region[0][1] <= center_y <= region[1][1]

# Function to count objects entering the region
def count_objects_in_region(boxes, region, counted_objects):
    """
    Count objects entering a specific region.
    :param boxes: List of detection boxes [(x1, y1, x2, y2, obj_id)].
    :param region: The region to check, as a tuple [(x1, y1), (x2, y2)].
    :param counted_objects: Set of objects already counted in the region.
    :return: Updated counted_objects set and count of new objects entering the region.
    """
    count = 0
    for box in boxes:
        x1, y1, x2, y2, obj_id = box

        # Check if the object is inside the region and hasn't been counted before
        if is_in_region(x1, y1, x2, y2, region) and obj_id not in counted_objects:
            counted_objects.add(obj_id)  # Mark this object as counted
            count += 1

    return counted_objects, count

# Function to dynamically set the detection region based on video dimensions
def get_center_region(video_width, video_height):
    """
    Calculate the center region of the video, with width being half of the video dimensions
    and height being 80% of the video height.
    :param video_width: Width of the video.
    :param video_height: Height of the video.
    :return: Region as a tuple [(x1, y1), (x2, y2)].
    """
    region_width = video_width // 2  # Width is 50% of the video width
    region_height = int(video_height * 0.9)  # Height is 80% of the video height

    # Coordinates for the region centered in the video
    x1 = (video_width - region_width) // 2
    y1 = (video_height - region_height) // 2
    x2 = x1 + region_width
    y2 = y1 + region_height

    return [(x1, y1), (x2, y2)]


# Example usage
if __name__ == '__main__':
    # Example: Open a video to get its dimensions
    video_path = 'video/litchi1.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video dimensions
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video dimensions: {video_width}x{video_height}")

    # Get the center detection region based on video dimensions
    region = get_center_region(video_width, video_height)
    print(f"Center region coordinates: {region}")

    # Example list of detected boxes [(x1, y1, x2, y2, obj_id)]
    detected_boxes = [
        (100, 200, 150, 250, 1),  # Example detection for object 1
        (400, 300, 450, 350, 2),  # Example detection for object 2
        (150, 200, 200, 250, 1),  # Object 1, already counted
    ]

    # Set to keep track of already counted object IDs
    counted_objects = set()

    # Count new objects entering the region
    counted_objects, count = count_objects_in_region(detected_boxes, region, counted_objects)

    print(f"Objects counted in the center region: {count}")
    print(f"Current counted object IDs: {counted_objects}")

    cap.release()
