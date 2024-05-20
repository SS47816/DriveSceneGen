import glob
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

####---initial config---####


def get_image_histogram(img):
    dx = img[0, :, :]
    dy = img[1, :, :]

    ## flatten the image
    dx_gray_values = dx.flatten()
    dy_gray_values = dy.flatten()

    ## compute the histogram
    dx_histogram, dxbins = np.histogram(dx_gray_values, bins=256, range=(0, 1))
    dy_histogram, dybins = np.histogram(dy_gray_values, bins=256, range=(0, 1))

    ## find the maximum peak of the histogram
    dx_max_gray_value = dxbins[np.argmax(dx_histogram)]
    dy_max_gray_value = dybins[np.argmax(dy_histogram)]

    ## plot the histogram
    # fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    # ax0.bar(dxbins[:-1], dx_histogram, width=1/256, align='edge', color='gray')
    # ax0.set_title('dx')

    # ax1.bar(dybins[:-1], dy_histogram, width=1/256, align='edge', color='gray')
    # ax1.set_title('dy')

    # ax2.imshow(dx,cmap='gray')
    # ax3.imshow(dy,cmap='gray')

    # plt.tight_layout()
    # plt.show()

    return dx_max_gray_value, dy_max_gray_value


def verify_vehicle(img, x, y):
    gradient = [0, 0, 0]

    dx_max_gray_value, dy_max_gray_value = get_image_histogram(img)

    # compute the gradient
    r = 2
    dx_grey = (
        1
        - img[
            0,
            max(0, y - r) : min(img.shape[1], y + r + 1),
            max(0, x - r) : min(img.shape[2], x + r + 1),
        ].mean()
    )
    dy_grey = img[
        1,
        max(0, y - r) : min(img.shape[1], y + r + 1),
        max(0, x - r) : min(img.shape[2], x + r + 1),
    ].mean()
    vel = img[
        2,
        max(0, y - r) : min(img.shape[1], y + r + 1),
        max(0, x - r) : min(img.shape[2], x + r + 1),
    ].mean()

    if (
        abs(dx_grey - dx_max_gray_value) > 0.05
        or abs(dy_grey - dy_max_gray_value) > 0.05
    ):
        gradient[0] = vel - 0.5
        gradient[1] = dx_grey - dx_max_gray_value
        gradient[2] = dy_grey - dy_max_gray_value

    return gradient


def estimate_agent_yaw(center: tuple, lanes: list) -> float:
    min_dist_waypoints = []
    for lane in lanes:
        dists = np.hypot(lane[:, 0] - center[0], lane[:, 1] - center[1])
        min_id = np.argmin(dists)
        min_dist_waypoints.append(
            (
                lane[min_id, 0],
                lane[min_id, 1],
                np.arctan2(lane[min_id, 4], lane[min_id, 3]),
                dists[min_id],
            )
        )

    min_dist_waypoints = np.array(min_dist_waypoints).reshape((-1, 4))
    min_id = np.argmin(min_dist_waypoints[:, -1].flatten())
    yaw = min_dist_waypoints[min_id, 2]
    dist = min_dist_waypoints[min_id, -1]

    return yaw, dist


def tansform_to_world_frame(agent: list, map_center: tuple, map_scale: float):
    # point: [x, y, z, dx, dy, dz]
    # agent: `list` [center_x, center_y, center_z, length, width, height, angle, velocity_x, velocity_y]
    agent[0] = agent[0] * map_scale - map_center[0]  # center_x
    agent[1] = map_center[1] - agent[1] * map_scale  # center_y
    agent[2] = agent[2] * map_scale  # center_z
    agent[3] = agent[3] * map_scale  # length
    agent[4] = agent[4] * map_scale  # width
    agent[5] = agent[5] * map_scale  # height
    agent[6] = agent[6] * (-1)  # angle
    agent[7] = agent[7]  # velocity_x
    agent[8] = agent[8] * (-1)  # velocity_y
    return agent


def normalize_angle_rad(angle) -> float:
    # normalize angle to (-pi, pi]
    while angle > np.pi:
        angle = angle - 2 * np.pi
    while angle <= -np.pi:
        angle = angle + 2 * np.pi
    return angle


def extract_agents(raw_img: np.ndarray, lanes: np.ndarray, map_range: float = 80.0, dist_thresh: float = 3.0, min_speed: float = 2.0, max_speed: float = 10.0):
    img_shape = raw_img.shape[1:]
    map_scale = map_range / img_shape[0]  # m/pixel
    map_center = (img_shape[0] / 2 * map_scale, img_shape[1] / 2 * map_scale)

    vehicles_list = []
    toimage = torch.cat(
        (raw_img[2:, :, :], raw_img[2:, :, :], raw_img[2:, :, :]), dim=0
    )

    ## convert the image to numpy array
    image = toimage.numpy()
    image = np.transpose(image, (1, 2, 0))
    img = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 100, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    for i, cnt in enumerate(contours):
        center, size, angle = cv2.minAreaRect(cnt)
        length, width = size

        if width < length:
            yaw = normalize_angle_rad(np.deg2rad(angle) + np.pi)
        else:
            yaw = normalize_angle_rad(np.deg2rad(angle) + np.pi / 2)
            width, length = length, width

        if length < 4.0/map_scale or width < 1.75/map_scale:
            continue
        length = min(length, 5.0/map_scale)
        width = min(width, 2.2/map_scale)
        height = 1.0/map_scale

        ## verify if the vehicle has the correct orientation and shape
        # Compute the velocity
        gradient = verify_vehicle(raw_img, int(center[0]), int(center[1]))
        # angle = np.arctan2(gradient[2], gradient[1])
        velocity = np.abs(gradient[0])*60.0

        ## export the properties of the vehicles
        ## [x, y, z, height, width, length, angle, velocity_x, velocity_y, area, [contour]]
        agent = tansform_to_world_frame(
            [
                center[0],
                center[1],
                0.0,
                length,
                width,
                height,
                yaw,
                velocity * np.cos(yaw),
                velocity * np.sin(yaw),
            ],
            map_center,
            map_scale,
        )
        # Find the vehicle yaw direction:
        lane_yaw, dist = estimate_agent_yaw(agent[:2], lanes)
        if dist < dist_thresh:
            velocity = min(velocity, max_speed)
            velocity = max(min_speed, velocity)
        else:
            velocity = 0.0
            
        agent[-3] = lane_yaw  # yaw
        agent[-2] = velocity * np.cos(lane_yaw)  # velocity_x
        agent[-1] = velocity * np.sin(lane_yaw)  # velocity_y

        vehicles_list.append(agent)

    return vehicles_list
