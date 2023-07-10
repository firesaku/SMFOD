from scipy.spatial import KDTree
# from dadapy.data import Data
import warnings
import numpy as np
from treelib import Tree
import dataPreprocessing
import skdim
from diptest import diptest

warnings.filterwarnings('ignore')
from sklearn.datasets import make_blobs
import math
from sklearn.decomposition import PCA
import copy


class Slime():
    def __init__(self, point_order=None, energy=1, tube_num=1, mucus_num=1, information=0, status=0,
                 last_move_distance=1):
        self.point_order = point_order
        self.energy = energy
        self.tube_num = tube_num
        self.mucus_num = mucus_num
        self.information = information
        self.status = status
        self.last_move_distance = last_move_distance


class Point():
    def __init__(self, attribute=None, sleeped_slime_num=0, tube=1, mucus=1000, rho=0, neighbor_list=None,
                 distance_list=None, max_k=0, passed_time=1):
        self.attribute = attribute
        self.sleeped_slime_num = sleeped_slime_num
        self.tube = tube
        self.mucus = mucus
        self.rho = rho
        self.neighbor_list = neighbor_list
        self.distance_list = distance_list
        self.max_k = max_k
        self.passed_time = passed_time


def find_max_k(data, Dthr, point_list):
    """
    自适应分配每个点对应的k近邻
    :param data: 输入数据
    :param Dthr: 判断阈值。由于密度差符合卡方分布，Dthr越大判断结果越符合点i所拥有的最大近邻。Dtrh对应置信度，因此该条件不是自由变量
    :param point_list: 点列表
    :return: 返回每个点对应的k近邻 密度rou 预测误差error 亮度light 近邻距离矩阵distances 近邻矩阵indices
    """
    # 计算点的内在维度
    # data_twoNN = Data(data)
    # data_twoNN.compute_distances(data.shape[0])
    # data_twoNN.compute_id_2NN()
    # id=data_twoNN.intrinsic_dim
    danco = skdim.id.MOM().fit(data)
    id = danco.dimension_
    # id = 2
    # 构建KD树
    tree = KDTree(data)
    distances, indices = tree.query(data, k=data.shape[0])
    dissimilarity = np.power(distances, id)
    V_matrix = np.diff(dissimilarity, axis=1)

    # 初始化每个点对应的k, 密度，预测误差，亮度
    list_k = [-1] * len(data)
    list_rou = [-1] * len(data)
    list_error = [-1] * len(data)
    list_light = [-1] * len(data)
    for i in range(len(data)):  # 遍历每一个点
        Dk_flag = False  # 判断是否有点满足密度差条件
        now_k = 0  # 当前近邻数
        while True:
            now_k += 1
            # 计算now_k
            j = indices[i][now_k]  # 找到当前点的第k个近邻； indices[i][0]为i点自身
            # 计算Dk 和 Dk1
            Dk = -2 * now_k * (np.log(np.sum(V_matrix[i][:now_k])) + np.log(np.sum(V_matrix[j][:now_k])) - 2 * np.log(
                np.sum(V_matrix[i][:now_k]) + np.sum(V_matrix[j][:now_k])) + np.log(4))
            Dk1 = -2 * now_k * (
                    np.log(np.sum(V_matrix[i][:now_k + 1])) + np.log(np.sum(V_matrix[j][:now_k + 1])) - 2 * np.log(
                np.sum(V_matrix[i][:now_k + 1]) + np.sum(V_matrix[j][:now_k + 1])) + np.log(4))
            if Dk < Dthr:  # 判断是否达到过阈值
                Dk_flag = True
            if ((Dk1 >= Dthr) and (Dk_flag == True)) or (
                    now_k == data.shape[0] - 1) == True:  # 如果【达到阈值】 或者 【遍历到最大近邻数】 则停止遍历
                list_k[i] = now_k
                list_rou[i] = now_k / np.sum(V_matrix[i][:now_k])
                list_error[i] = np.sqrt((4 * now_k + 2) / ((now_k - 1) * now_k))
                point_list[i].rho = np.log(list_rou[i]) / list_error[i]
                point_list[i].neighbor_list = indices[i]
                point_list[i].distance_list = distances[i]
                point_list[i].max_k = now_k
                break

    return point_list


def Model(X, alpha, beta, gamma, Dthr, slime_num, T, contamination, seed):
    """
    史莱姆孤立点检测
    :param X: 数据集
    :param alpha: 控制管道粗细
    :param beta: 控制粘液浓度
    :param gamma: 控制史莱姆生命值
    :param Dthr: 控制自适应近邻阈值
    :param slime_num: 控制史莱姆数量
    :param T: 算法执行轮数
    :return: 孤立度对应的孤立点列表

    1. 先计算每个点的密度
    2. 然后将点根据密度大小，从高到低排序
    3. 随机撒入史莱姆
    4. 遍历每一只史莱姆
    5. 遍历当前史莱姆的每一个近邻
    6. 通过密度进行轮盘赌，让史莱姆选择下一步要去的方向
    7. 史莱姆移动后，目标点的管道、黏液浓度要更新，史莱姆的生命值也会更新。史莱姆的生命值会影响粘液和管道的含量
        7.1 如果史莱姆的移动距离过远，使得自己移动后生命值为0，则史莱姆脱水休眠
        7.2 如果史莱姆移动的新点上有休眠的史莱姆，则会增加该史莱姆的信息，距离表现为扩大近邻搜索范围
    8. 黏液越多越不愿意去，黏液越少越愿意去。移动距离越大，则越会增加黏液
    9. 管道越多，越愿意去，密度越大管道越多
    10. 黏液叠得快，管道叠得慢
    """
    np.random.seed(seed)
    # 1. 初始化点和史莱姆
    point_list = []
    slime_list = []
    random_order = np.random.randint(0, X.shape[0], size=slime_num)
    for i in range(0, X.shape[0]):
        point_list.append(Point(attribute=np.array(X[i])))
    for i in range(0, len(random_order)):
        slime_list.append(Slime(point_order= random_order[i]))
        point_list[random_order[i]].passed_time += 1

    find_max_k(X, Dthr, point_list)
    unpassed_list=[] # 改动1：增加了一个列表，用来存储没有被走过的点，一开始的值是所有点
    for i in range(0,X.shape[0]):
        unpassed_list.append(i)
    t = 0
    while t <= T:
        for slime in slime_list:
            if slime.status == 1:
                continue
            # ---------史莱姆选点------------
            now_point_order = copy.copy(slime.point_order)
            now_k_range = min(len(point_list), point_list[now_point_order].max_k + slime.information)
            temp_info_list = []
            neighbor_order_list = []
            jump_flag=True# 改动二：增加史莱姆是否要跳出近邻范围的判断标识。如果跳出了近邻范围，那么就去走没有走过的点
            next_neighbor_order_for_now_point = 0
            for k in range(1, now_k_range):
                neighbor_order = point_list[now_point_order].neighbor_list[k]
                neighbor_order_list.append(neighbor_order)
                now_info = (point_list[neighbor_order].mucus  - point_list[neighbor_order].rho - point_list[neighbor_order].tube)
                temp_info_list.append(now_info)
                if point_list[neighbor_order].passed_time<=1: # 如果近邻里有点没有被走过，史莱姆则不跳出该近邻范围
                    jump_flag=False
            if (jump_flag==True) and (len(unpassed_list)!=0): # 如果史莱姆可以跳出近邻范围，并且有点没有被走过，那么史莱姆就选没被走过的点
                next_point=unpassed_list[0]
            else:
                next_point=neighbor_order_list[np.argmax(temp_info_list)]
            #--X-------史莱姆选点----------X--
            if next_point in unpassed_list: # 表示移除已经被走过的点
                unpassed_list.remove(next_point)

            # --------史莱姆移动------------
            move_distance = point_list[now_point_order].distance_list[next_neighbor_order_for_now_point+1]
            slime.point_order = next_point
            old_energy = copy.copy(slime.energy)
            slime.energy = slime.energy + gamma * (point_list[now_point_order].rho - move_distance )
            print(t,"轮史莱姆从",now_point_order,"移动到",next_point,"能量从",old_energy,"变为",slime.energy,"距离为",move_distance,"上一轮距离",slime.last_move_distance )
            if slime.last_move_distance == move_distance:
                slime.information += 1
            slime.last_move_distance = move_distance
            if old_energy > slime.energy:
                slime.information += 1
            point_list[next_point].passed_time += 1
            if slime.energy <= 0:
                slime.status = 1
                point_list[next_point].sleeped_slime_num += 1

            point_list[next_point].tube += alpha * point_list[next_point].passed_time
            point_list[next_point].mucus -= beta * move_distance
            if point_list[next_point].sleeped_slime_num > 0:
                slime.information += point_list[next_point].sleeped_slime_num * point_list[next_point].max_k
            # --X-----史莱姆移动-----------X--
        t += 1

    outlier_tube_list = []
    outlier_mucus_list = []
    outlier_passed_time_list = []
    outlier_degree = []
    outlier_x = []
    outlier_y = []
    
    for point in point_list:
        outlier_tube_list.append(point.tube)
        outlier_mucus_list.append(point.mucus/point.passed_time)
        outlier_passed_time_list.append(point.passed_time)
        outlier_degree.append((np.log(point.mucus)/point.passed_time)/point.tube)
        outlier_x.append(np.log(point.mucus)/point.passed_time)
        outlier_y.append(point.tube)

    dataPreprocessing.DecisionPlot(outlier_x,outlier_y)
    
    indices = [i for i, _ in sorted(enumerate(outlier_degree), key=lambda x: x[1], reverse=True)]
    label=[0]*X.shape[0]
    outlier_num=int(contamination * X.shape[0])
    for i in range(0,outlier_num):
        label[indices[i]]=1
    return outlier_degree,label,indices
