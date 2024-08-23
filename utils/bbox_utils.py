def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    
    # Return the center of the bounding box for both x and y coordinates
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]