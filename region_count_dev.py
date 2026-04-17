import time

# Initialize a dictionary to store object entry frame counts
object_entry_frame = {}

def count_objects_in_region(boxes, region, counted_objects, frame_threshold=20):
    """
    Count objects entering the region and track their IDs, with time window filtering using frame threshold.
    :param boxes: List of detection boxes [x1, y1, x2, y2, obj_id]
    :param region: The region to check, as a tuple [(x1, y1), (x2, y2)].
    :param counted_objects: Set of objects already counted.
    :param frame_threshold: Minimum number of frames an object must stay in the region to be counted.
    :return: Updated counted_objects set and current region count.
    """
    region_count = 0  # Initialize region count

    for box in boxes:
        # Only unpack the first five elements, ignore conf
        x1, y1, x2, y2, obj_id = box

        # Check if object has entered the region
        if obj_id not in counted_objects and is_object_in_region(x1, y1, x2, y2, region):
            # Check if this object has already entered the region recently
            if obj_id not in object_entry_frame:
                object_entry_frame[obj_id] = 1  # Record that the object entered the region on the current frame
            else:
                # Calculate the number of frames the object has been in the region
                object_entry_frame[obj_id] += 1
                if object_entry_frame[obj_id] >= frame_threshold:
                    counted_objects.add(obj_id)
                    region_count += 1
                    del object_entry_frame[obj_id]  # Object has been counted, remove from tracking

    return counted_objects, region_count


def is_object_in_region(x1, y1, x2, y2, region):
    """
    Check if the object's bounding box is inside the defined region.
    :param x1, y1, x2, y2: Coordinates of the bounding box.
    :param region: The region to check, as a tuple [(x1, y1), (x2, y2)].
    :return: True if the object is inside the region, False otherwise.
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return region[0][0] <= center_x <= region[1][0] and region[0][1] <= center_y <= region[1][1]

def get_center_region(video_width, video_height, region_ratio=(0.5, 0.9)):
    """
    Calculate the center region of the video, with width and height being specific ratios of the video dimensions.
    :param video_width: Width of the video.
    :param video_height: Height of the video.
    :param region_ratio: Tuple to define the region size (width_ratio, height_ratio)
    :return: Region as a tuple [(x1, y1), (x2, y2)].
    """
    region_width = int(video_width * region_ratio[0])  # Width is 50% of the video width
    region_height = int(video_height * region_ratio[1])  # Height is 80% of the video height

    # Coordinates for the region centered in the video
    x1 = (video_width - region_width) // 2
    y1 = (video_height - region_height) // 2
    x2 = x1 + region_width
    y2 = y1 + region_height

    return [(x1, y1), (x2, y2)]

# Example usage:

if __name__ == "__main__":
    # Example video dimensions
    video_width = 1280
    video_height = 720

    # Define the region
    region = get_center_region(video_width, video_height)

    # Example list of detected boxes [(x1, y1, x2, y2, obj_id, conf)]
    detected_boxes = [
        (100, 200, 150, 250, 1, 0.9),  # Example detection for object 1
        (400, 300, 450, 350, 2, 0.85),  # Example detection for object 2
        (150, 200, 200, 250, 1, 0.95),  # Object 1, already counted
    ]

    # Set to keep track of already counted object IDs
    counted_objects = set()

    # Count objects in region (using 50 frames as the threshold)
    counted_objects, region_count = count_objects_in_region(detected_boxes, region, counted_objects)

    print(f"Objects in region: {region_count}")
    print(f"Region coordinates: {region}")
    print(f"Counted object IDs: {counted_objects}")
