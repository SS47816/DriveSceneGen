import heapq
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

current_num = 0

####---initial config---####

class Point:
    def __init__(self, y, x, orientation, prev=None):
        self.y = y
        self.x = x
        self.orientation = orientation
        self.prev = prev

    def __lt__(self, other):
        if self.x < other.x:
            return True
        elif self.x > other.x:
            return False
        else:
            return self.y < other.y

def combine_dx_dy(dx: float, dy: float, mx: float, my: float, thresh: float=0.1) -> int:
    if np.fabs(dx - mx) <= thresh and np.fabs(dy - my) <= thresh:
        return 0
    else:
        return 255
    

def get_gray_image(img_color: Image, plot: bool=False) -> Image:
    img_tensor = np.array(img_color, dtype=float)

    x_channel = img_tensor[:, :, 0]
    y_channel = img_tensor[:, :, 1]
    z_channel = img_tensor[:, :, 2]

    ## flatten the image
    dx_gray_values = x_channel.flatten()
    dy_gray_values = y_channel.flatten()
    dz_gray_values = z_channel.flatten()

    ## compute the histogram
    dx_histogram, dxbins = np.histogram(dx_gray_values, bins=256, range=(0, 1))
    dy_histogram, dybins = np.histogram(dy_gray_values, bins=256, range=(0, 1))
    dz_histogram, dzbins = np.histogram(dz_gray_values, bins=256, range=(0, 1))

    ## find the maximum peak of the histogram
    dx_max_gray_value = dxbins[np.argmax(dx_histogram)]
    dy_max_gray_value = dybins[np.argmax(dy_histogram)]
    dz_max_gray_value = dybins[np.argmax(dz_histogram)]

    ## compute the mean of the maximum peak
    mx = dx_max_gray_value
    my = dy_max_gray_value
    mz = dz_max_gray_value
    # print(mx, my, mz)

    img_mask = [combine_dx_dy(dx, dy, mx, my) for (dx, dy) in zip(dx_gray_values, dy_gray_values)]
    img_mask_tensor = np.array(img_mask, dtype=np.uint8).reshape(img_tensor.shape[:2])
    img_gray_tensor = np.stack((img_mask_tensor, img_mask_tensor, img_mask_tensor), axis=-1)
    # print(img_mask_tensor.shape)
    img_gray = Image.fromarray(img_gray_tensor)

    ## plot the histogram
    if plot:
        fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, sharex='row', sharey='row')
        ax0.bar(dxbins[:-1], dx_histogram, width=1/256, align='edge', color='r')
        ax0.set_title('x_channel')

        ax1.bar(dybins[:-1], dy_histogram, width=1/256, align='edge', color='g')
        ax1.set_title('y_channel')

        ax2.bar(dzbins[:-1], dz_histogram, width=1/256, align='edge', color='b')
        ax2.set_title('z_channel')

        ax3.imshow(x_channel, cmap='Reds')
        ax4.imshow(y_channel, cmap='Greens')
        ax5.imshow(z_channel, cmap='Blues')

        plt.tight_layout()
        plt.show()

    return img_mask_tensor

def generate_centerline(img,polyline_keypoints):
    center_line = np.zeros(img.shape[1:])
        
    for i in range(polyline_keypoints.shape[0]):
        x = polyline_keypoints[i][0]
        y = polyline_keypoints[i][1]

        if (x < img.shape[1]-1 and x > 0) and (y  < img.shape[2]-1 and y > 0):
            center_line[int(y)][int(x)] = 1

    # add additional channel for converting to RGB image
    center_line = torch.tensor(center_line)
    # plt.imshow(center_line,cmap='gray')
    # plt.show()

    return center_line

def generate_polyline_keypoints(center_line, dx, dy):
    height, width = center_line.shape

    polyline_keypoints = []
    visited_points = set()  # 记录已访问过的点的坐标

    for y in range(height):
        for x in range(width):
            if center_line[y][x] > 0.99 and (x, y) not in visited_points:
                orientation = np.arctan2(dx[y][x]-0.5, dy[y][x]-0.5)
                polyline_keypoints.append([x, y, orientation])
                visited_points.add((x, y))
            elif (x, y) in visited_points:
                print("Error: duplicated points",(x, y))

    return torch.tensor(polyline_keypoints)

def compare_neighbor(current_point, neighbors, neighbor_visited, visited):
    nearest_neighbor = None
    min_orientation_diff = float('inf')  # 初始设定一个较大的值

    for neighbor in neighbors:
        if neighbor not in neighbor_visited and neighbor not in visited:
            
            # 逐个查询本点到邻点的方向变化趋势
            neighbor_orientation_dx = neighbor.x - current_point.x
            neighbor_orientation_dy = neighbor.y - current_point.y
            neighbor_orientation = np.arctan2(neighbor_orientation_dy, neighbor_orientation_dx)

            neighbor_orientation_diff = neighbor.orientation - current_point.orientation
            if abs(neighbor_orientation_diff) > math.radians(270):
                neighbor_orientation_diff = abs(neighbor_orientation_diff) - math.radians(270)
                # print("neighbor_orientation_diff",math.degrees(neighbor_orientation_diff))

            # 获得上一个点朝向和方向的变化趋势
            if current_point.prev:
                prev_orientation_diff = current_point.orientation - current_point.prev.orientation

                # 获得上一个点到本点的方向变化趋势
                prev_orientation_dx = current_point.x - current_point.prev.x
                prev_orientation_dy = current_point.y - current_point.prev.y
                prev_orientation = np.arctan2(prev_orientation_dy, prev_orientation_dx)
                if abs(prev_orientation) > math.radians(270):
                    prev_orientation = abs(prev_orientation) - math.radians(270)
                    # print("prev_orientation",math.degrees(prev_orientation))
            else:
                prev_orientation_diff = 0
                prev_orientation = 0
            
            # 用于判断下一个点朝向和方向变化是否与上一个点朝向和方向变化是否一致
            diff_product = prev_orientation_diff * neighbor_orientation_diff 
            ori_product = prev_orientation * neighbor_orientation
            ori_diff = abs(prev_orientation - neighbor_orientation)# - math.radians(90)
            if ori_diff > math.radians(270):
                ori_diff = abs(ori_diff - math.radians(270))
                # print("ori_diff",math.degrees(ori_diff))
            
            # 查找最近的邻居点，要求符合：
            # 1."neighbor.orientation"与"current_point.orientation"的差最小
            # 2."neighbor.orientation - current_point.orientation"和"current_point.orientation - current_point.prev.orientation"差值同向 
            # 3."arctan(neighbor.xy - current_point.xy)"和"arctan(current_point.xy - current_point,prev.xy)"的方向变化也要一致 
            # 3."arctan(neighbor.xy - current_point.xy)"和"arctan(current_point.xy - current_point,prev.xy)"的方向差值小于90度
            # 4.邻点不是本点
            if (abs(neighbor_orientation_diff) < min_orientation_diff) and (diff_product >= 0) and (ori_product >=0) and (ori_diff <= math.radians(90)) and (neighbor.x != current_point.x or neighbor.y != current_point.y):
                min_orientation_diff = abs(neighbor_orientation_diff)
                nearest_neighbor = neighbor

            neighbor_visited.add(neighbor)

    if nearest_neighbor:
        return nearest_neighbor
    else:
        return None

def find_neighbours(current_point, points, visited):
    neighbors = []

    for point in points:

        neighbor_orientation_diff = point.orientation - current_point.orientation
        if abs(neighbor_orientation_diff) > np.radians(330):
            neighbor_orientation_diff = abs(neighbor_orientation_diff) - np.radians(330)
        
        ## filter the points that are too close to the current point
        if abs(point.x - current_point.x) <= 1 and abs(point.y - current_point.y) <= 1 and point not in visited:
            visited.add(point)  # 将"当前节点"加入已访问节点集合

        ## 将"当前节点"附近的4个像素距离，且方向差值小于30度的点加入到"邻居节点集合"中
        if abs(point.x - current_point.x) <= 3 and abs(point.y - current_point.y) <= 3 and abs(neighbor_orientation_diff) < math.radians(30) and point not in visited:
            
            neighbors.append(point)

    ## 按照neighbors和current_point的距离从小到大排序
    neighbors = sorted(neighbors, key=lambda p: (abs(p.x - current_point.x) + abs(p.y - current_point.y)))

    return neighbors

def explore(current_point, points, visited, paths, max_point_count=1500):
    global current_num # 用于记录当前path中已经搜索点的数量
    neighbor_visited = set() # 用于记录"当前点的邻居节点"是否已经被访问过
    visited.add(current_point)  # 将"当前节点"加入已访问节点集合

    neighbors = find_neighbours(current_point, points, visited) # 获取"当前节点"附近的邻居节点

    nearest_neighbor = compare_neighbor(current_point, neighbors, neighbor_visited, visited) # 将"当前节点"与所有"邻居"相比，获取"当前节点"的最近邻居节点

    # 如果符合条件的最近邻居节点存在，且当前path中的点的数量小于max_point_count
    if nearest_neighbor and current_num < max_point_count:
        current_num += 1
        nearest_neighbor.prev = current_point  # 设置前驱节点
        visited.add(nearest_neighbor)
        explore(nearest_neighbor, points, visited, paths) # 继续搜索
    else:  # 如果当前节点没有可访问的邻居节点，或到达了线段的末端
        path = []
        node = current_point
        while node:
            path.append([node.y, node.x, node.orientation])
            node = node.prev
        
        ## only save path with more than 1 points
        ## reverse path if the orientation of the first two points is not consistent with the orientation of the dxdy
        if len(path) > 1:
            i = len(path)//2
            x_g = path[i][0]-path[i-1][0]
            y_g = path[i][1]-path[i-1][1]
            ori_diff = abs(path[i-1][2]-np.arctan2(x_g, y_g))

            if ori_diff > math.radians(270):
                ori_diff = ori_diff - math.radians(270)

            if ori_diff > math.radians(90):
                path.reverse()
                # print(abs(math.degrees(path_ori - (path[0][2]+path[1][2])/2)))

            current_num = 0
            paths.append(torch.tensor(path))

def search(points_tensor,paths):
    # 将输入张量转换为 Point 对象的列表
    points = [Point(y, x, orientation) for y, x, orientation in points_tensor]

    # 创建一个优先队列并将所有点添加进去
    queue = [(p.x, p.y, p.orientation, p) for p in points]
    heapq.heapify(queue)  # 构造优先队列

    visited = set()

    while queue:
        _, _, _, current_point = heapq.heappop(queue)  # 弹出x和y都最小的点
        if current_point in visited:
            continue
        explore(current_point, points, visited, paths)

    return paths

def drop_wrong_points(paths):
    new_paths = []
    dropped_path = []

    for i, path in enumerate(paths):
        if len(path) > 1:
            x_g = np.array([(path[i+1,0]-path[i,0])/10 for i in range(len(path[:,0])-1)])
            y_g = np.array([(path[i+1,1]-path[i,1])/10 for i in range(len(path[:,1])-1)])
            x_g = np.append(x_g,x_g[-1])
            y_g = np.append(y_g,y_g[-1])
            gradient = np.arctan2(y_g, x_g)

            x_o = np.array([math.sin(path[i,2])/5 for i in range(len(path[:,0]))])
            y_o = np.array([math.cos(path[i,2])/5 for i in range(len(path[:,1]))])
            orientation = np.arctan2(y_o, x_o)

            diff = abs(gradient - orientation)

            for j, point in enumerate(path):
                
                if diff[j] > np.radians(270):
                        diff[j] = abs(diff[j]-np.radians(270))
                if diff[j]>np.radians(90):
                    dropped_path.append(path)
                    # print("drop wrong points",np.degrees(diff[j]))
                    if j > len(path)//2:
                        paths[i] = path[:j-1]
                    else:
                        paths[i] = path[j+1:]
                    
                    break

    for i in range(len(paths)):
        if len(paths[i]) > 1:
            new_paths.append(paths[i])

    return new_paths

def concat_short_paths(paths):

    # concat paths
    del_list = []
    additonal_path = []
    path_used = set()

    for i, pi in enumerate(paths):
        if len(pi) < 10:
            for j, pj in enumerate(paths):
                # calculate the distance, gradient and orientation difference between the start point and the end point of each path
                p_i = np.array([pi[-1][0],pi[-1][1]])
                p_j = np.array([pj[0][0],pj[0][1]])
                x_distance = abs(pj[0][0]-pi[-1][0])
                y_distance = abs(pj[0][1]-pi[-1][1])
                gradient = np.arctan2(p_j[0]-p_i[0], p_j[1]-p_i[1])
                gradient_diff = abs(gradient-pi[-1][2])
                ori_diff = abs(pj[0][2]-pi[-1][2])

                if gradient_diff > np.radians(300):
                    gradient_diff = abs(gradient_diff-np.radians(300))
                if ori_diff > np.radians(300):
                    ori_diff = abs(ori_diff-np.radians(300))

                if x_distance < 5 and y_distance < 5 and gradient_diff < np.radians(60) and ori_diff < np.radians(15) and i!=j:
                    if i not in path_used:
                        new_path = torch.cat((pi,pj),dim=0)
                        if i not in del_list:
                            del_list.append(i)
                        if j not in del_list:
                            del_list.append(j)
                        additonal_path.append(new_path)
                        path_used.add(i)

    paths = [p for i, p in enumerate(paths) if i not in del_list]
    
    for path in additonal_path:
        paths.append(path)

    ## delete the path with less than 4 points
    # path_del_list = []
    # for i, path in enumerate(paths):
    #     if path.shape[0] < 4:
    #         path_del_list.append(i)

    # paths = [p for i, p in enumerate(paths) if i not in path_del_list]

    return paths

def compute_dxdy_from_ori(paths):
    new_paths = []

    for i, path in enumerate(paths):
        new_path = []
        for j, point in enumerate(path):

            ## path:[x y z dx dy dz]
            new_path.append([point[0],point[1],0,np.sin(point[2]),np.cos(point[2]),0])

        new_paths.append(new_path)
    
    return new_paths

def tansform_to_world_frame(path: list, center: tuple, scale: float):
    # path: [point1, point2, ...]
    # point: [x, y, z, dx, dy, dz]
    path_world_frame = []
    for pt in path:
        pt[0] = pt[0] * scale - center[0]   # x
        pt[1] = pt[1] * scale - center[1]   # y
        pt[2] = 0.0                         # z
        pt[3] = pt[3] * scale               # dx
        pt[4] = pt[4] * scale               # dy
        pt[5] = 0.0                         # dz
        path_world_frame.append(pt)
    
    return path_world_frame

def random_color():
    return '#{:02X}{:02X}{:02X}'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_random_colors(n):
    return [random_color() for _ in range(n)]

def extract_centerlines(img_tensor: np.ndarray, map_range: float=80.0, plot_lane: bool=False):

    paths = []

    ## extract lanes from 3 channels img (centerline, dx, dy)
    # centerline = img_tensor[0,:,:]
    # dx = 1 - img_tensor[1,:,:]
    # dy = img_tensor[2,:,:]
    # polyline_keypoints = generate_polyline_keypoints(centerline, dx, dy)

    # resize = transforms.Resize((256, 256),antialias=False)
    image = img_tensor.permute(1,2,0)
    dx = 1 - image[:,:,0]
    dy = image[:,:,1]
    
    to_tensor = transforms.ToTensor()
    centerline_raw = to_tensor(get_gray_image(image,plot=False)).squeeze(0).numpy()
    centerline = torch.from_numpy(skeletonize(centerline_raw))

    polyline_keypoints = generate_polyline_keypoints(centerline, dx, dy)

    ## search paths
    paths = search(polyline_keypoints,paths)
    
    ## drop wrong points
    paths = drop_wrong_points(paths)

    ## concat short paths
    paths = concat_short_paths(paths)

    paths = compute_dxdy_from_ori(paths)

    if plot_lane==True:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(centerline, cmap='gray')        
        ax1.invert_yaxis()
        color = generate_random_colors(len(paths))
        for index, path in enumerate(paths):

            path = np.array(path)
            
            ## only plot path points
            ax2.plot(path[:, 0], path[:, 1], linewidth=1)

            ## plot arrow according to path sequence
            x_g = np.array([(path[i+1,0]-path[i,0])/10 for i in range(len(path[:,0])-1)])
            y_g = np.array([(path[i+1,1]-path[i,1])/10 for i in range(len(path[:,1])-1)])
            x_g = np.append(x_g,0)
            y_g = np.append(y_g,0)
            ax2.quiver(path[:, 0], path[:, 1], x_g, y_g, angles='xy', scale_units='xy', scale=0.1)

            ## plot arrow according to path orientation
            # x_g = np.array([math.sin(path[i,2])/5 for i in range(len(path[:,0]))])
            # y_g = np.array([math.cos(path[i,2])/5 for i in range(len(path[:,1]))])
            # ax2.quiver(path[:, 0], path[:, 1], x_g, y_g, color=color[index], angles='xy', scale_units='xy', scale=0.1)

        plt.show()

    img_shape = img_tensor.shape[1:]
    scale = map_range / img_shape[0] # m/pixel
    center = (img_shape[0]/2 * scale, img_shape[1]/2 * scale)
    paths_world_frame = [tansform_to_world_frame(path, center, scale) for path in paths]

    return paths_world_frame


