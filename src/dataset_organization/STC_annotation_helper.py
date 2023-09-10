import cv2

def get_area(segmentation_img):
    area = segmentation_img.shape[0] * segmentation_img.shape[1]

    return area

def get_keypoints(instance_id, annotation_data):
    """
    Return:
        1) 
            Number of All keypoints

        2)
            Segmentation coordinates  [[x1, y1, v1, x2, y2, v2, ...]]
            v is "visibility":
                v=0: Not labeled
                v=1: labeled but not visible
                v=2: labeled and visible
            If there is an object splitting the person in half, there would be two lists:
            [[x1, y1, v1, ...], [x2, y2, v2, ...]]
            TODO: I havent seen any images like that in the solo dataset, so I don't know how to handle it.
    """
    for d in annotation_data:
        type = d['@type']
        if type == "type.unity.com/unity.solo.KeypointAnnotation":
            keypoint_info = d
            break

    values = keypoint_info['values']
    for val in values:
        if val['instanceId'] == instance_id:
            all_keypoints = val['keypoints']

            num_keypoints = len(all_keypoints)
            
            final_keypoint_arr = []
            for pt in all_keypoints:
                location = pt['location']
                x = location[0]
                y = location[1]
                v = pt['state']

                key_pt_iteration = [x, y, v]
                final_keypoint_arr.extend(key_pt_iteration)
            
            return num_keypoints, final_keypoint_arr

def get_segmentation_data(instance_id, segmentation_img):
    """
    Return:
        Segmentation coordinates  [[x1, y1, x2, y2, ...]]
        If there is an object splitting the person in half, there would be two lists:
        [[x1,y1,x2,y2,...],[x3,y3,x4,y4,...]]
        TODO: I havent seen any images like that in the solo dataset, so I don't know how to handle it.
        TODO: actually implement from segmentation img, segmentation data is not in the json file
                (it technically is, but it has nothing useful)
    """
    return [[None]]

def get_bbox_data(instance_id, annotation_data):
    """
    Return:
        Bounding box info [x, y, width, height]
        where (0,0) is the top left corner of the image and x,y is the top left corner of the bbox
    """
    for d in annotation_data:
        type = d['@type']
        if type == "type.unity.com/unity.solo.BoundingBox2DAnnotation":
            bbox_info = d
            break
    
    values = bbox_info['values']

    for val in values:
        if val['instanceId'] == instance_id:
            origin = val['origin']
            dimension = val['dimension']

            x = origin[0]
            y = origin[1]
            width = dimension[0]
            height = dimension[1]

            return [x, y, width, height]
        

def get_image_fov_data(frame_data):
    """
    Returns:
        FOV data
        Focal Length data
    """
    metrics = frame_data['metrics']

    i=0
    for d in metrics:
        type = d['@type']
        if type == "CameraFOV":
            fov_data = d
            i+=1
        elif type == "CameraFocalLength":
            focal_length_data = d
            i+=1
        if i>=2:
            break

    return fov_data['value'], focal_length_data['value']