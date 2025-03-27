import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse
import mmcv
from mmcv import Config
import matplotlib.transforms as transforms
from mmdet3d.datasets import build_dataset
import cv2
import torch
import numpy as np
from PIL import Image
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from scipy.spatial import ConvexHull
from PIL import Image
import cv2
import imageio
import math
from tracking.cmap_utils.match_utils import *

Label2Name = {
    0: "ped_crossing",
    1: "divider",
    2: "boundary",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize groundtruth and results")
    parser.add_argument("config", help="config file path")
    parser.add_argument(
        "--dataset_name",
        help="dataset name",
        default="nuScenes",
        choices=["nuScenes", "argoverse2"],
    )
    parser.add_argument(
        "--simplify", default=0.2, type=float, help="Line simplification tolerance"
    )
    parser.add_argument("--line_opacity", default=0.75, type=float, help="Line opacity")

    args = parser.parse_args()

    return args


def merge_corssing(polylines):
    convex_hull_polygon = find_largest_convex_hull(polylines)
    return convex_hull_polygon


def find_largest_convex_hull(polylines):
    # Merge all points from the polylines into a single collection
    all_points = []
    for polyline in polylines:
        all_points.extend(list(polyline.coords))

    # Convert the points to a NumPy array for processing with scipy
    points_array = np.array(all_points)

    # Compute the convex hull using scipy
    hull = ConvexHull(points_array)

    # Extract the vertices of the convex hull
    hull_points = points_array[hull.vertices]

    # Create a shapely Polygon object representing the convex hull
    convex_hull_polygon = LineString(hull_points).convex_hull

    return convex_hull_polygon


def project_point_onto_line(point, line):
    """Project a point onto a line segment and return the projected point."""
    line_start, line_end = np.array(line.coords[0]), np.array(line.coords[1])
    line_vec = line_end - line_start
    point_vec = np.array(point.coords[0]) - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    t = np.clip(t, 0.0, 1.0)
    nearest = line_start + t * line_vec
    return Point(nearest)


def find_nearest_projection_on_polyline(point, polyline):
    """Find the nearest projected point of a point onto a polyline."""
    min_dist = float("inf")
    nearest_point = None
    for i in range(len(polyline.coords) - 1):
        segment = LineString(polyline.coords[i : i + 2])
        proj_point = project_point_onto_line(point, segment)
        dist = point.distance(proj_point)
        if dist < min_dist:
            min_dist = dist
            nearest_point = proj_point
    return np.array(nearest_point.coords)


def find_and_sort_intersections(segmenet1, segment2):
    # Convert polylines to LineString objects

    # Find the intersection between the two LineStrings
    intersection = segmenet1.intersection(segment2)

    # Prepare a list to store intersection points
    intersections = []

    # Check the type of intersection
    if "Point" in intersection.geom_type:
        # Single point or multiple points
        if intersection.geom_type == "MultiPoint":
            intersections.extend(list(intersection))
        else:
            intersections.append(intersection)
    elif "LineString" in intersection.geom_type:
        # In case of lines or multiline, get boundary points (start and end points of line segments)
        if intersection.geom_type == "MultiLineString":
            for line in intersection:
                intersections.extend(list(line.boundary))
        else:
            intersections.extend(list(intersection.boundary))

    # Remove duplicates and ensure they are Point objects
    unique_intersections = [
        Point(coords) for coords in set(pt.coords[0] for pt in intersections)
    ]

    # Sort the intersection points by their distance along the first polyline
    sorted_intersections = sorted(
        unique_intersections, key=lambda pt: segmenet1.project(pt)
    )

    return sorted_intersections


def get_intersection_point_on_line(line, intersection):
    intersection_points = find_and_sort_intersections(LineString(line), intersection)
    if len(intersection_points) >= 2:
        line_intersect_start = intersection_points[0]
        line_intersect_end = intersection_points[-1]
    elif len(intersection_points) == 1:
        if intersection.contains(Point(line[0])):
            line_intersect_start = Point(line[0])
            line_intersect_end = intersection_points[0]
        elif intersection.contains(Point(line[-1])):
            line_intersect_start = Point(line[-1])
            line_intersect_end = intersection_points[0]
        else:
            return None, None
    else:
        return None, None
    return line_intersect_start, line_intersect_end


def merge_l2_points_to_l1(line1, line2, line2_intersect_start, line2_intersect_end):
    # get nearest point on line2 to line2_intersect_start
    line2_point_to_merge = []
    line2_intersect_start_dis = line2.project(line2_intersect_start)
    line2_intersect_end_dis = line2.project(line2_intersect_end)
    for point in np.array(line2.coords):
        point_geom = Point(point)
        dis = line2.project(point_geom)
        if dis > line2_intersect_start_dis and dis < line2_intersect_end_dis:
            line2_point_to_merge.append(point)

    # merged the points
    merged_line2_points = []
    for point in line2_point_to_merge:
        # Use the `project` method to find the distance along the polyline to the closest point
        point_geom = Point(point)
        # Use the `interpolate` method to find the actual point on the polyline
        closest_point_on_line = find_nearest_projection_on_polyline(point_geom, line1)
        if len(closest_point_on_line) == 0:
            merged_line2_points.append(point)
        else:
            merged_line2_points.append(((closest_point_on_line + point) / 2)[0])

    if len(merged_line2_points) == 0:
        merged_line2_points = np.array([]).reshape(0, 2)
    else:
        merged_line2_points = np.array(merged_line2_points)

    return merged_line2_points


def segment_line_based_on_merged_area(line, merged_points):

    if len(merged_points) == 0:
        return np.array(line.coords), np.array([]).reshape(0, 2)

    first_merged_point = merged_points[0]
    last_merged_point = merged_points[-1]

    start_dis = line.project(Point(first_merged_point))
    end_dis = line.project(Point(last_merged_point))

    start_segmenet = []
    for point in np.array(line.coords):
        point_geom = Point(point)
        if line.project(point_geom) < start_dis:
            start_segmenet.append(point)

    end_segmenet = []
    for point in np.array(line.coords):
        point_geom = Point(point)
        if line.project(point_geom) > end_dis:
            end_segmenet.append(point)

    if len(start_segmenet) == 0:
        start_segmenet = np.array([]).reshape(0, 2)
    else:
        start_segmenet = np.array(start_segmenet)

    if len(end_segmenet) == 0:
        end_segmenet = np.array([]).reshape(0, 2)
    else:
        end_segmenet = np.array(end_segmenet)

    return start_segmenet, end_segmenet


def get_bbox_size_for_points(points):
    if len(points) == 0:
        return 0, 0

    # Initialize min and max coordinates with the first point
    min_x, min_y = points[0]
    max_x, max_y = points[0]

    # Iterate through each point to update min and max coordinates
    for x, y in points[1:]:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return max_x - min_x, max_y - min_y


def get_longer_segmenent_to_merged_points(
    l1_segment, l2_segment, merged_line2_points, segment_type="start"
):
    # remove points from segments if it's too close to merged_line2_points
    l1_segment_temp = []
    if len(merged_line2_points) > 1:
        merged_polyline = LineString(merged_line2_points)
        for point in l1_segment:
            if merged_polyline.distance(Point(point)) > 0.1:
                l1_segment_temp.append(point)
    elif len(merged_line2_points) == 1:
        for point in l1_segment:
            if Point(point).distance(Point(merged_line2_points[0])) > 0.1:
                l1_segment_temp.append(point)
    elif len(merged_line2_points) == 0:
        l1_segment_temp = l1_segment

    l1_segment = np.array(l1_segment_temp)

    l2_segmenet_temp = []
    if len(merged_line2_points) > 1:
        merged_polyline = LineString(merged_line2_points)
        for point in l2_segment:
            if merged_polyline.distance(Point(point)) > 0.1:
                l2_segmenet_temp.append(point)
    elif len(merged_line2_points) == 1:
        for point in l2_segment:
            if Point(point).distance(Point(merged_line2_points[0])) > 0.1:
                l2_segmenet_temp.append(point)
    elif len(merged_line2_points) == 0:
        l2_segmenet_temp = l2_segment

    l2_segment = np.array(l2_segmenet_temp)

    if segment_type == "start":

        temp = l1_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[0])

        l1_start_box_size = get_bbox_size_for_points(temp)

        temp = l2_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[0])
        l2_start_box_size = get_bbox_size_for_points(temp)

        if (
            l2_start_box_size[0] * l2_start_box_size[1]
            >= l1_start_box_size[0] * l1_start_box_size[1]
        ):
            longer_segment = l2_segment
        else:
            longer_segment = l1_segment
    else:
        temp = l1_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[-1])
        l1_end_box_size = get_bbox_size_for_points(temp)

        temp = l2_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[-1])
        l2_end_box_size = get_bbox_size_for_points(temp)

        if (
            l2_end_box_size[0] * l2_end_box_size[1]
            >= l1_end_box_size[0] * l1_end_box_size[1]
        ):
            longer_segment = l2_segment
        else:
            longer_segment = l1_segment

    if len(longer_segment) == 0:
        longer_segment = np.array([]).reshape(0, 2)
    else:
        longer_segment = np.array(longer_segment)

    return longer_segment


def get_line_lineList_max_intersection(merged_lines, line, thickness=4):
    pre_line = merged_lines[-1]
    max_iou = 0
    merged_line_index = 0
    for line_index, one_merged_line in enumerate(merged_lines):
        line1 = LineString(one_merged_line)
        line2 = LineString(line)
        thick_line1 = line1.buffer(thickness)
        thick_line2 = line2.buffer(thickness)
        intersection = thick_line1.intersection(thick_line2)
        if intersection.area / thick_line2.area > max_iou:
            max_iou = intersection.area / thick_line2.area
            pre_line = np.array(line1.coords)
            merged_line_index = line_index
    return intersection, pre_line, merged_line_index


def algin_l2_with_l1(line1, line2):

    if len(line1) > len(line2):
        l2_len = len(line2)
        line1_geom = LineString(line1)
        interval_length = line1_geom.length / (l2_len - 1)
        line1 = [
            np.array(line1_geom.interpolate(interval_length * i)) for i in range(l2_len)
        ]

    elif len(line1) < len(line2):
        l1_len = len(line1)
        line2_geom = LineString(line2)
        interval_length = line2_geom.length / (l1_len - 1)
        line2 = [
            np.array(line2_geom.interpolate(interval_length * i)) for i in range(l1_len)
        ]

    # make line1 and line2 same direction, pre_line.coords[0] shold be closer to line2.coords[0]
    line1_geom = LineString(line1)
    line2_flip = np.flip(line2, axis=0)

    line2_traj_len = 0
    for point_idx, point in enumerate(line2):
        line2_traj_len += np.linalg.norm(point - line1[point_idx])

    flip_line2_traj_len = 0
    for point_idx, point in enumerate(line2_flip):
        flip_line2_traj_len += np.linalg.norm(point - line1[point_idx])

    if abs(flip_line2_traj_len - line2_traj_len) < 3:
        # get the trajectory length
        line2_walk_len = 0
        for point in line2:
            point_geom = Point(point)
            proj_point = find_nearest_projection_on_polyline(point_geom, line1_geom)
            if len(proj_point) != 0:
                line2_walk_len += line1_geom.project(Point(proj_point[0]))

        flip_line2_walk_len = 0
        for point in line2:
            point_geom = Point(point)
            proj_point = find_nearest_projection_on_polyline(point_geom, line1_geom)
            if len(proj_point) != 0:
                flip_line2_walk_len += line1_geom.project(Point(proj_point[0]))

        if flip_line2_walk_len < line2_walk_len:
            return line2_flip
        else:
            return line2

    if flip_line2_traj_len < line2_traj_len:
        return line2_flip
    else:
        return line2


def _is_u_shape(line, direction):
    assert direction in ["left", "right"], "Wrong direction argument {}".format(
        direction
    )
    line_geom = LineString(line)
    length = line_geom.length
    mid_point = np.array(line_geom.interpolate(length / 2).coords)[0]
    start = line[0]
    end = line[-1]

    if direction == "left":
        cond1 = mid_point[0] < start[0] and mid_point[0] < end[0]
    else:
        cond1 = mid_point[0] > start[0] and mid_point[0] > end[0]

    dist_start_end = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
    cond2 = length >= math.pi / 2 * dist_start_end

    return cond1 and cond2


def check_circle(pre_line, vec):

    # if the last line in merged_lines is a circle
    if np.linalg.norm(pre_line[0] - pre_line[-1]) == 0:
        return True

    # if the last line in merged_lines is almost a circle and the new line is close to the circle
    if np.linalg.norm(pre_line[0] - pre_line[-1]) < 0.1:
        vec_2_circle_distance = 0
        for point in vec:
            vec_2_circle_distance += LineString(pre_line).distance(Point(point))
        if vec_2_circle_distance < 3:
            return True
    return False


def connect_polygon(merged_polyline, merged_lines):
    start_end_connect = [merged_polyline[0], merged_polyline[-1]]
    iou = []
    length_ratio = []
    for one_merged_line in merged_lines:
        line1 = LineString(one_merged_line)
        line2 = LineString(start_end_connect)
        thickness = 1
        thick_line1 = line1.buffer(thickness)
        thick_line2 = line2.buffer(thickness)
        intersection = thick_line1.intersection(thick_line2)
        iou.append(intersection.area / thick_line2.area)
        length_ratio.append(line1.length / line2.length)

    if max(iou) > 0.95 and max(length_ratio) > 3.0:
        merged_polyline = np.concatenate(
            (merged_polyline, [merged_polyline[0]]), axis=0
        )
    return merged_polyline


def iou_merge_boundry(merged_lines, vec, thickness=1):

    # intersection : the intersection area between the new line and the line in the merged_lines; is a polygon
    intersection, pre_line, merged_line_index = get_line_lineList_max_intersection(
        merged_lines, vec, thickness
    )

    # corner case: check if the last line in merged_lines is a circle
    if check_circle(pre_line, vec):
        return merged_lines

    # Handle U-shape, the main corner case
    if _is_u_shape(pre_line, "left"):
        if _is_u_shape(vec, "right"):
            # Two u shapes with opposite directions, directly generate a polygon exterior
            polygon = find_largest_convex_hull([LineString(pre_line), LineString(vec)])
            merged_lines[-1] = np.array(polygon.exterior.coords)
            return merged_lines
        elif not _is_u_shape(vec, "left"):
            line_geom1 = LineString(pre_line)
            line1_dists = np.array([line_geom1.project(Point(x)) for x in pre_line])
            split_mask = line1_dists > line_geom1.length / 2
            split_1 = LineString(pre_line[~split_mask])
            split_2 = LineString(pre_line[split_mask])

            # get the projected distance
            np1 = np.array(
                nearest_points(split_1, Point(Point(pre_line[-1])))[0].coords
            )[0]
            np2 = np.array(
                nearest_points(split_2, Point(Point(pre_line[0])))[0].coords
            )[0]
            dist1 = np.linalg.norm(np1 - pre_line[-1])
            dist2 = np.linalg.norm(np2 - pre_line[0])
            dist = min(dist1, dist2)

            if dist < thickness:
                line_geom2 = LineString(vec)
                dist1 = line_geom2.distance(Point(pre_line[0]))
                dist2 = line_geom2.distance(Point(pre_line[-1]))
                pt = pre_line[0] if dist1 <= dist2 else pre_line[-1]
                if vec[0][0] > vec[1][0]:
                    vec = np.array(vec[::-1])
                    line_geom2 = LineString(vec)
                proj_length = line_geom2.project(Point(pt))
                l2_select_mask = np.array(
                    [line_geom2.project(Point(x)) > proj_length for x in vec]
                )
                selected_l2 = vec[l2_select_mask]
                merged_result = np.concatenate(
                    [pre_line[:-1, :], pt[None, ...], selected_l2], axis=0
                )
                merged_lines[-1] = merged_result
                return merged_lines

    # align the new line with the line in the merged_lines so that points on two lines are traversed in the same direction
    vec = algin_l2_with_l1(pre_line, vec)
    line1 = LineString(pre_line)
    line2 = LineString(vec)

    # get the intersection points between IOU area and two lines
    line1_intersect_start, line1_intersect_end = get_intersection_point_on_line(
        pre_line, intersection
    )
    line2_intersect_start, line2_intersect_end = get_intersection_point_on_line(
        vec, intersection
    )

    # If no intersection points are found, use the last point of the line1 and the first point of the line2 as the intersection points --> this is a corner case that we will connect the two lines head to tail directly
    if (
        line1_intersect_start is None
        or line1_intersect_end is None
        or line2_intersect_start is None
        or line2_intersect_end is None
    ):
        line1_intersect_start = Point(pre_line[-1])
        line1_intersect_end = Point(pre_line[-1])
        line2_intersect_start = Point(vec[0])
        line2_intersect_end = Point(vec[0])

    # merge the points on line2's intersection area towards line1
    merged_line2_points = merge_l2_points_to_l1(
        line1, line2, line2_intersect_start, line2_intersect_end
    )
    # merge the points on line1's intersection area towards line2
    merged_line1_points = merge_l2_points_to_l1(
        line2, line1, line1_intersect_start, line1_intersect_end
    )

    # segment the lines based on the merged points (intersection area); split the line in to start segment and merged segment and end segment
    l2_start_segment, l2_end_segment = segment_line_based_on_merged_area(
        line2, merged_line2_points
    )
    l1_start_segment, l1_end_segment = segment_line_based_on_merged_area(
        line1, merged_line1_points
    )

    # choose the longer segment between line1 and line2 to be the final start segment and end segment
    start_segment = get_longer_segmenent_to_merged_points(
        l1_start_segment, l2_start_segment, merged_line2_points, segment_type="start"
    )
    end_segment = get_longer_segmenent_to_merged_points(
        l1_end_segment, l2_end_segment, merged_line2_points, segment_type="end"
    )
    merged_polyline = np.concatenate(
        (start_segment, merged_line2_points, end_segment), axis=0
    )

    # corner case : check if need to connect the polyline to form a circle
    merged_polyline = connect_polygon(merged_polyline, merged_lines)

    merged_lines[merged_line_index] = merged_polyline

    return merged_lines


def iou_merge_divider(merged_lines, vec, thickness=1):
    # intersection : the intersection area between the new line and the line in the merged_lines; is a polygon
    # pre_line : the line in merged_lines that has max IOU with the new line
    intersection, pre_line, merged_line_index = get_line_lineList_max_intersection(
        merged_lines, vec, thickness
    )
    # align the new line with the line in the merged_lines so that points on two lines are traversed in the same direction
    vec = algin_l2_with_l1(pre_line, vec)

    line1 = LineString(pre_line)
    line2 = LineString(vec)

    # get the intersection points between IOU area and two lines
    line1_intersect_start, line1_intersect_end = get_intersection_point_on_line(
        pre_line, intersection
    )
    line2_intersect_start, line2_intersect_end = get_intersection_point_on_line(
        vec, intersection
    )

    # If no intersection points are found, use the last point of the line1 and the first point of the line2 as the intersection points --> this is a corner case that we will connect the two lines head to tail directly
    if (
        line1_intersect_start is None
        or line1_intersect_end is None
        or line2_intersect_start is None
        or line2_intersect_end is None
    ):
        line1_intersect_start = Point(pre_line[-1])
        line1_intersect_end = Point(pre_line[-1])
        line2_intersect_start = Point(vec[0])
        line2_intersect_end = Point(vec[0])

    # merge the points on line2's intersection area towards line1
    merged_line2_points = merge_l2_points_to_l1(
        line1, line2, line2_intersect_start, line2_intersect_end
    )
    # merge the points on line1's intersection area towards line2
    merged_line1_points = merge_l2_points_to_l1(
        line2, line1, line1_intersect_start, line1_intersect_end
    )

    # segment the lines based on the merged points (intersection area); split the line in to start segment and merged segment and end segment
    l2_start_segment, l2_end_segment = segment_line_based_on_merged_area(
        line2, merged_line2_points
    )
    l1_start_segment, l1_end_segment = segment_line_based_on_merged_area(
        line1, merged_line1_points
    )

    # choose the longer segment between line1 and line2 to be the final start segment and end segment
    start_segment = get_longer_segmenent_to_merged_points(
        l1_start_segment, l2_start_segment, merged_line2_points, segment_type="start"
    )
    end_segment = get_longer_segmenent_to_merged_points(
        l1_end_segment, l2_end_segment, merged_line2_points, segment_type="end"
    )
    merged_polyline = np.concatenate(
        (start_segment, merged_line2_points, end_segment), axis=0
    )

    # update the merged_lines
    merged_lines[merged_line_index] = merged_polyline

    return merged_lines


def merge_divider(vecs=None, thickness=1):
    merged_lines = []
    for vec in vecs:

        # if the merged_lines is empty, add the first line
        if len(merged_lines) == 0:
            merged_lines.append(vec)
            continue

        # thicken the vec (the new line) and the merged_lines calculate the max IOU between the new line and the merged_lines
        iou = []
        for one_merged_line in merged_lines:
            line1 = LineString(one_merged_line)
            line2 = LineString(vec)
            thick_line1 = line1.buffer(thickness)
            thick_line2 = line2.buffer(thickness)
            intersection = thick_line1.intersection(thick_line2)
            iou.append(intersection.area / thick_line2.area)

        # If the max IOU is 0, add the new line to the merged_lines
        if max(iou) == 0:
            merged_lines.append(vec)
        # If IOU is not 0, merge the new line with the line in the merged_lines
        else:
            merged_lines = iou_merge_divider(merged_lines, vec, thickness=thickness)

    return merged_lines


def merge_boundary(vecs=None, thickness=1, iou_threshold=0.95):
    merged_lines = []
    for vec in vecs:

        # if the merged_lines is empty, add the first line
        if len(merged_lines) == 0:
            merged_lines.append(vec)
            continue

        # thicken the vec (the new line) and the merged_lines calculate the max IOU between the new line and the merged_lines
        iou = []
        for one_merged_line in merged_lines:
            line1 = LineString(one_merged_line)
            line2 = LineString(vec)
            thick_line1 = line1.buffer(thickness)
            thick_line2 = line2.buffer(thickness)
            intersection = thick_line1.intersection(thick_line2)
            iou.append(intersection.area / thick_line2.area)

        # If the max IOU larger than the threshold, skip the new line
        if max(iou) > iou_threshold:
            continue

        # If IOU is not 0, merge the new line with the line in the merged_lines
        if max(iou) > 0:
            merged_lines = iou_merge_boundry(merged_lines, vec, thickness=thickness)
        else:
            merged_lines.append(vec)

    return merged_lines


def get_consecutive_vectors_with_opt(
    prev_vectors=None,
    prev2curr_matrix=None,
    origin=None,
    roi_size=None,
    denormalize=False,
    clip=False,
):
    # transform prev vectors
    prev2curr_vectors = dict()
    for label, vecs in prev_vectors.items():
        if len(vecs) > 0:
            vecs = np.stack(vecs, 0)
            vecs = torch.tensor(vecs)
            N, num_points, _ = vecs.shape
            if denormalize:
                denormed_vecs = vecs * roi_size + origin  # (num_prop, num_pts, 2)
            else:
                denormed_vecs = vecs
            denormed_vecs = torch.cat(
                [
                    denormed_vecs,
                    denormed_vecs.new_zeros((N, num_points, 1)),  # z-axis
                    denormed_vecs.new_ones((N, num_points, 1)),  # 4-th dim
                ],
                dim=-1,
            )  # (num_prop, num_pts, 4)

            transformed_vecs = torch.einsum(
                "lk,ijk->ijl", prev2curr_matrix, denormed_vecs.double()
            ).float()
            normed_vecs = (
                transformed_vecs[..., :2] - origin
            ) / roi_size  # (num_prop, num_pts, 2)
            if clip:
                normed_vecs = torch.clip(normed_vecs, min=0.0, max=1.0)
            prev2curr_vectors[label] = normed_vecs
        else:
            prev2curr_vectors[label] = vecs

    # convert to ego space for visualization
    for label in prev2curr_vectors:
        if len(prev2curr_vectors[label]) > 0:
            prev2curr_vectors[label] = prev2curr_vectors[label] * roi_size + origin
    return prev2curr_vectors


def get_prev2curr_vectors(
    vecs=None,
    prev2curr_matrix=None,
    origin=None,
    roi_size=None,
    denormalize=False,
    clip=False,
):
    # transform prev vectors
    if len(vecs) > 0:
        vecs = np.stack(vecs, 0)
        vecs = torch.tensor(vecs)
        N, num_points, _ = vecs.shape
        if denormalize:
            denormed_vecs = vecs * roi_size + origin  # (num_prop, num_pts, 2)
        else:
            denormed_vecs = vecs
        denormed_vecs = torch.cat(
            [
                denormed_vecs,
                denormed_vecs.new_zeros((N, num_points, 1)),  # z-axis
                denormed_vecs.new_ones((N, num_points, 1)),  # 4-th dim
            ],
            dim=-1,
        )  # (num_prop, num_pts, 4)

        transformed_vecs = torch.einsum(
            "lk,ijk->ijl", prev2curr_matrix, denormed_vecs.double()
        ).float()
        vecs = (transformed_vecs[..., :2] - origin) / roi_size  # (num_prop, num_pts, 2)
        if clip:
            vecs = torch.clip(vecs, min=0.0, max=1.0)
        # vecs = vecs * roi_size + origin

    return vecs


def plot_fig_merged_per_frame(
    num_frames,
    id_prev2curr_pred_vectors,
    id_prev2curr_pred_frame,
    args,
):

    # key the current status of the instance, add into the dict when it first appears
    instance_bank = dict()

    # plot the figure at each frame
    merged_maps = []
    for frame_timestep in range(num_frames):
        merged_map = []

        for vec_tag, vec_all_frames in id_prev2curr_pred_vectors.items():
            vec_frame_info = id_prev2curr_pred_frame[vec_tag]
            first_appear_frame = sorted(list(vec_frame_info.keys()))[0]

            # 看当前帧是否有这个instance
            need_merge = False
            if frame_timestep < first_appear_frame:  # the instance has not appeared
                continue
            elif frame_timestep in vec_frame_info:
                need_merge = True
                vec_index_in_instance = vec_frame_info[frame_timestep]

            label, vec_glb_idx = vec_tag.split("_")
            label = int(label)
            vec_glb_idx = int(vec_glb_idx)

            # 如果需要merge，就把当前帧的vector加入到instance_bank中
            if need_merge:
                curr_vec = vec_all_frames[vec_index_in_instance]
                curr_vec_polyline = LineString(curr_vec)
                if vec_tag not in instance_bank:  # if the instance first appears
                    polylines = [
                        curr_vec_polyline,
                    ]
                else:  # if the instance has appeared before, polylines = previous merged polyline + current polyline
                    polylines = instance_bank[vec_tag] + [
                        curr_vec_polyline,
                    ]
            else:  # if the instance has not appeared in this frame
                polylines = instance_bank[vec_tag]

            if label == 0:  # crossing, merged by convex hull
                if need_merge:
                    # 融合多个polyline
                    polygon = merge_corssing(polylines)
                    polygon = polygon.simplify(args.simplify)
                    vector = np.array(polygon.exterior.coords)
                else:  # if no new instance, use the previous merged polyline to plot
                    vector = np.array(polylines[0].coords)

                pts = vector[:, :2]

                # update instance bank for ped
                updated_polyline = LineString(vector)
                instance_bank[vec_tag] = [
                    updated_polyline,
                ]
                merged_map.append(
                    {"cls_name": Label2Name[label], "geom": pts, "type": "vectorized"}
                )

            elif label == 1:  # divider, merged fitting a polyline
                if need_merge:
                    polylines_vecs = [
                        np.array(one_line.coords) for one_line in polylines
                    ]
                    polylines_vecs = merge_divider(polylines_vecs)
                else:  # if no new instance, use the previous merged polyline to plot
                    polylines_vecs = [np.array(line.coords) for line in polylines]

                polylines_vecs = [vec for vec in polylines_vecs if vec.shape[0] > 1]
                for one_line in polylines_vecs:
                    one_line = np.array(
                        LineString(one_line).simplify(args.simplify * 2).coords
                    )
                    pts = one_line[:, :2]

                    merged_map.append(
                        {
                            "cls_name": Label2Name[label],
                            "geom": pts,
                            "type": "vectorized",
                        }
                    )

                # update instance bank for line
                updated_polylines = [LineString(vec) for vec in polylines_vecs]
                instance_bank[vec_tag] = updated_polylines

            elif label == 2:  # boundary, do not merge
                if need_merge:
                    polylines_vecs = [
                        np.array(one_line.coords) for one_line in polylines
                    ]
                    polylines_vecs = merge_boundary(polylines_vecs)
                else:  # if no new instance, use the previous merged polyline to plot
                    polylines_vecs = [np.array(line.coords) for line in polylines]

                polylines_vecs = [vec for vec in polylines_vecs if vec.shape[0] > 1]
                for one_line in polylines_vecs:
                    one_line = np.array(
                        LineString(one_line).simplify(args.simplify).coords
                    )
                    pts = one_line[:, :2]

                    merged_map.append(
                        {
                            "cls_name": Label2Name[label],
                            "geom": pts,
                            "type": "vectorized",
                        }
                    )

                # update instance bank for line
                updated_polylines = [LineString(vec) for vec in polylines_vecs]
                instance_bank[vec_tag] = updated_polylines

        merged_maps.append(merged_map)

    return merged_maps


def vis_pred_data(
    scene_name="", pred_results=None, origin=None, roi_size=None, args=None
):

    # get the item index of the scene
    index_list = []
    for index in range(len(pred_results)):
        if pred_results[index]["scene_name"] == scene_name:
            index_list.append(index)

    id_prev2curr_pred_vectors = defaultdict(list)
    id_prev2curr_pred_frame_info = defaultdict(list)
    id_prev2curr_pred_frame = defaultdict(list)

    # iterate through each frame
    last_index = index_list[-1]
    pred_scene_data = defaultdict(list)
    for index in index_list:

        vectors = np.array(pred_results[index]["vectors"]).reshape(
            (len(np.array(pred_results[index]["vectors"])), 20, 2)
        )
        if abs(vectors.max()) <= 1:
            curr_vectors = vectors * roi_size + origin
        else:
            curr_vectors = vectors

        # get the transformation matrix of the last frame
        prev_e2g_trans = torch.tensor(
            pred_results[index]["meta"]["ego2global_translation"], dtype=torch.float64
        )
        prev_e2g_rot = torch.tensor(
            pred_results[index]["meta"]["ego2global_rotation"], dtype=torch.float64
        )
        curr_e2g_trans = torch.tensor(
            pred_results[last_index]["meta"]["ego2global_translation"],
            dtype=torch.float64,
        )
        curr_e2g_rot = torch.tensor(
            pred_results[last_index]["meta"]["ego2global_rotation"], dtype=torch.float64
        )
        prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
        prev_e2g_matrix[:3, :3] = prev_e2g_rot
        prev_e2g_matrix[:3, 3] = prev_e2g_trans

        curr_g2e_matrix = torch.eye(4, dtype=torch.float64)
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix

        pred_scene_data["trajectory"].append(prev2curr_matrix)
        frame = []
        for label, vecs in zip(pred_results[index]["labels"], curr_vectors):
            frame.append(
                {"cls_name": Label2Name[label], "geom": vecs, "type": "vectorized"}
            )
        pred_scene_data["frames"].append(frame)

        prev2curr_pred_vectors = get_prev2curr_vectors(
            curr_vectors, prev2curr_matrix, origin, roi_size, False, False
        )
        prev2curr_pred_vectors = prev2curr_pred_vectors * roi_size + origin

        # vecs = np.stack(curr_vectors, 0)
        # vecs = torch.tensor(vecs)
        # N, num_points, _ = vecs.shape
        # denormed_vecs = vecs
        # denormed_vecs = torch.cat(
        #     [
        #         denormed_vecs,
        #         denormed_vecs.new_zeros((N, num_points, 1)),  # z-axis
        #         denormed_vecs.new_ones((N, num_points, 1)),  # 4-th dim
        #     ],
        #     dim=-1,
        # )  # (num_prop, num_pts, 4)

        # prev2curr_pred_vectors2 = torch.einsum(
        #     "lk,ijk->ijl", prev2curr_matrix, denormed_vecs.double()
        # ).float()[..., :2]

        for i, (label, vec_glb_idx) in enumerate(
            zip(pred_results[index]["labels"], pred_results[index]["global_ids"])
        ):
            dict_key = "{}_{}".format(label, vec_glb_idx)
            id_prev2curr_pred_vectors[dict_key].append(prev2curr_pred_vectors[i])
            id_prev2curr_pred_frame_info[dict_key].append(
                [
                    pred_results[index]["local_idx"],
                    len(id_prev2curr_pred_frame[dict_key]),
                ]
            )

        for key, frame_info in id_prev2curr_pred_frame_info.items():
            frame_localIdx = dict()
            for frame_time, local_index in frame_info:
                frame_localIdx[frame_time] = local_index
            id_prev2curr_pred_frame[key] = frame_localIdx

    # print(f"roi_size: ")
    # import matplotlib
    # matplotlib.use("TkAgg")
    # fig, ax = plt.subplots()
    # for frame, pose in zip(pred_scene_data["frames"], pred_scene_data["trajectory"]):
    #     vecs = np.stack([vec["geom"] for vec in frame], 0)
    #     vecs = torch.tensor(vecs)
    #     N, num_points, _ = vecs.shape
    #     denormed_vecs = vecs
    #     denormed_vecs = torch.cat(
    #         [
    #             denormed_vecs,
    #             denormed_vecs.new_zeros((N, num_points, 1)),  # z-axis
    #             denormed_vecs.new_ones((N, num_points, 1)),  # 4-th dim
    #         ],
    #         dim=-1,
    #     )
    #     transformed_vecs = torch.einsum(
    #         "lk,ijk->ijl", pose, denormed_vecs.double()
    #     ).float()[..., :2]
    #     for i, vec in enumerate(transformed_vecs):
    #         ax.plot(vec[:, 0], vec[:, 1])
    # plt.show()
    # print(f"roi_size: {roi_size}")

    # sort the id_prev2curr_pred_vectors
    id_prev2curr_pred_vectors = {
        key: id_prev2curr_pred_vectors[key] for key in sorted(id_prev2curr_pred_vectors)
    }

    num_frames = len(index_list)
    merged_maps = plot_fig_merged_per_frame(
        num_frames,
        id_prev2curr_pred_vectors,  # 记录每个instance所有的vector（已经转换到全局坐标系）
        id_prev2curr_pred_frame,  # 记录每个instance，每一次出现的帧序号
        args,
    )
    pred_scene_data["merged_maps"] = merged_maps
    pred_scene_data["scene_name"] = scene_name
    return pred_scene_data


def vis_gt_data(scene_name, args, dataset, gt_data, origin, roi_size):

    gt_info = gt_data[scene_name]
    gt_info_list = []
    ids_info = []

    # get the item index of the sample
    for index, one_idx in enumerate(gt_info["sample_ids"]):
        gt_info_list.append(dataset[one_idx])
        ids_info.append(gt_info["instance_ids"][index])

    # key : label, vec_glb_idx ; value : list of vectors in the last frame's coordinate
    id_prev2curr_pred_vectors = defaultdict(list)
    # dict to store some information of the vectors
    id_prev2curr_pred_frame_info = defaultdict(list)
    # key : label, vec_glb_idx ; value : {frame_time : idx of the vector; idx range from 0 to the number of vectors of the same instance }
    id_prev2curr_pred_frame = defaultdict(dict)

    scene_len = len(gt_info_list)
    gt_scene_data = defaultdict(list)
    for idx in range(scene_len):
        curr_vectors = dict()
        # denormalize the vectors
        for label, vecs in gt_info_list[idx]["vectors"].data.items():
            if len(vecs) > 0:  # if vecs != []
                curr_vectors[label] = vecs * roi_size + origin
            else:
                curr_vectors[label] = vecs

        # get the transformation matrix of the last frame
        prev_e2g_trans = torch.tensor(
            gt_info_list[idx]["img_metas"].data["ego2global_translation"],
            dtype=torch.float64,
        )
        prev_e2g_rot = torch.tensor(
            gt_info_list[idx]["img_metas"].data["ego2global_rotation"],
            dtype=torch.float64,
        )
        curr_e2g_trans = torch.tensor(
            gt_info_list[scene_len - 1]["img_metas"].data["ego2global_translation"],
            dtype=torch.float64,
        )
        curr_e2g_rot = torch.tensor(
            gt_info_list[scene_len - 1]["img_metas"].data["ego2global_rotation"],
            dtype=torch.float64,
        )
        prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
        prev_e2g_matrix[:3, :3] = prev_e2g_rot
        prev_e2g_matrix[:3, 3] = prev_e2g_trans

        curr_g2e_matrix = torch.eye(4, dtype=torch.float64)
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

        # get the transformed vectors from current frame to the last frame
        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix

        gt_scene_data["trajectory"].append(prev2curr_matrix)
        gt_frame = []
        for label, vecs in curr_vectors.items():
            if len(vecs) < 1:
                continue
            for vec in vecs:
                gt_frame.append(
                    {"cls_name": Label2Name[label], "geom": vec, "type": "vectorized"}
                )
        gt_scene_data["gt_frames"].append(gt_frame)

        prev2curr_pred_vectors = get_consecutive_vectors_with_opt(
            curr_vectors, prev2curr_matrix, origin, roi_size, False, False
        )
        for label, id_info in ids_info[idx].items():
            for vec_local_idx, vec_glb_idx in id_info.items():
                dict_key = "{}_{}".format(label, vec_glb_idx)
                id_prev2curr_pred_vectors[dict_key].append(
                    prev2curr_pred_vectors[label][vec_local_idx]
                )
                # gt_info_list[idx]["seq_info"].data[1] stores the frame time that the vector appears
                id_prev2curr_pred_frame_info[dict_key].append(
                    [
                        gt_info_list[idx]["seq_info"].data[1],
                        len(id_prev2curr_pred_frame[dict_key]),
                    ]
                )  # set len(id_prev2curr_pred_frame[dict_key]) to be the index of the vector belongs to the same instance
        for key, frame_info in id_prev2curr_pred_frame_info.items():
            frame_localIdx = dict()
            for frame_time, local_index in frame_info:
                frame_localIdx[frame_time] = local_index
            id_prev2curr_pred_frame[key] = frame_localIdx

    # sort the id_prev2curr_pred_vectors by label and vec_glb_idx
    id_prev2curr_pred_vectors = {
        key: id_prev2curr_pred_vectors[key] for key in sorted(id_prev2curr_pred_vectors)
    }

    merged_maps = plot_fig_merged_per_frame(
        len(gt_info_list),
        id_prev2curr_pred_vectors,
        id_prev2curr_pred_frame,
        args,
    )
    gt_scene_data["gt_merged_maps"] = merged_maps
    gt_scene_data["scene_name"] = scene_name
    return gt_scene_data


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    dataset = build_dataset(cfg.match_config)

    scene_name2idx = {}
    scene_name2token = {}

    for idx, sample in enumerate(dataset.samples):
        scene = sample["scene_name"]
        if scene not in scene_name2idx:
            scene_name2idx[scene] = []
            scene_name2token[scene] = []
        scene_name2idx[scene].append(idx)

    # load the GT data
    # dataset_name = "nuscenes"
    dataset_name = "argoverse2"
    if dataset_name == "nuscenes":
        gt_data_path = "/home/qzj/datasets/nuscenes/custom/maptracker/nuscenes_map_infos_val_gt_tracks.pkl"
        gt_data = mmcv.load(gt_data_path)

        pred_data_path = "work_dirs/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions.pkl"
        pred_data = pickle.load(open(pred_data_path, "rb"))

        save_dir = "/home/qzj/datasets/nuscenes/custom/maptracker/mapping_results"
    elif dataset_name == "argoverse2":
        gt_data_path = "/home/qzj/datasets/argoverse2/sensor/custom/maptracker/av2_map_infos_val_gt_tracks.pkl"
        gt_data = mmcv.load(gt_data_path)

        pred_data_path = "work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions.pkl"
        pred_data = pickle.load(open(pred_data_path, "rb"))

        save_dir = (
            "/home/qzj/datasets/argoverse2/sensor/custom/maptracker/mapping_results"
        )

    all_scene_names = sorted(list(scene_name2idx.keys()))

    roi_size = torch.tensor(cfg.roi_size).numpy()
    origin = torch.tensor(cfg.pc_range[:2]).numpy()
    os.makedirs(save_dir, exist_ok=True)
    from tqdm import tqdm

    all_results = {}
    for scene_name in tqdm(all_scene_names):

        gt_scene_data = vis_gt_data(
            scene_name=scene_name,
            args=args,
            dataset=dataset,
            gt_data=gt_data,
            origin=origin,
            roi_size=roi_size,
        )

        pred_scene_data = vis_pred_data(
            scene_name=scene_name,
            pred_results=pred_data,
            origin=origin,
            roi_size=roi_size,
            args=args,
        )

        scene_data = {}
        scene_data.update(gt_scene_data)
        scene_data.update(pred_scene_data)

        save_path = os.path.join(save_dir, f"{scene_name}.pkl")
        mmcv.dump(scene_data, save_path)
        print(f"Scene {scene_name} saved to {save_path}")
        all_results[scene_name] = scene_data

    mmcv.dump(all_results, os.path.join(save_dir, "../all_results.pkl"))


if __name__ == "__main__":
    main()
