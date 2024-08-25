def get_bbox_center(bbox):
    # Get the center of the bounding box
    x1, y1, x2, y2 = bbox
    
    # Return the center of the bounding box for both x and y coordinates
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    # Return the width of the bounding box
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    # Calculate the Euclidean distance between two points
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5