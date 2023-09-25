from pandas import DataFrame, read_csv
from math import radians, sin, cos, asin, sqrt
# from geopy.distance import geodesic
import numpy as np
import pulp
from numba import jit
# from time import time
import json
from multiprocessing import Pool


# haversine算法用经纬度计算距离，误差比geopy大很多，但是速度非常快
@jit(nopython=True)
def haversine(lat1, lng1, lat2, lng2):
    rr = 6378.137  # rr为地球赤道半径，单位:千米
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2-lng1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2*asin(sqrt(a))*rr*1000
    distance = round(distance, 1)  # 单位:米
    return distance


# DTW算法，返回损失矩阵，损失矩阵右下角值越接近0，相似度越高
def DTW(a, b):
    n, m = len(a), len(b)
    DTWM = np.zeros((n+1, m+1))
    DTWM.fill(np.inf)
    DTWM[0, 0] = 0
    for i in range(n):
        for j in range(m):
            last_min = np.min([DTWM[i, j], DTWM[i+1, j], DTWM[i, j+1]])
            DTWM[i+1, j+1] = abs(a[i] - b[j]) + last_min
    return DTWM


# 计算航向差距
def deltaCourse(a, b):
    delta = a - b
    if delta < -180:
        delta = 360 + delta
    else:
        if delta > 180:
            delta = 360 - delta
    return abs(delta)


# 计算航向差距的DTW算法
def DTWcourse(a, b):
    n, m = len(a), len(b)
    DTWM = np.zeros((n+1, m+1))
    DTWM.fill(np.inf)
    DTWM[0, 0] = 0
    for i in range(n):
        for j in range(m):
            last_min = np.min([DTWM[i, j], DTWM[i+1, j], DTWM[i, j+1]])
            DTWM[i+1, j+1] = deltaCourse(a[i], b[j]) + last_min
    return DTWM

# 计算经纬度距离的DTW算法
def DTWdist(a, b):
    n, m = len(a), len(b)
    DTWM = np.zeros((n+1, m+1))
    DTWM.fill(np.inf)
    DTWM[0, 0] = 0
    for i in range(n):
        for j in range(m):
            # dist = geodesic((a[i, 1], a[i, 0]), (b[j, 1], b[j, 0])).meters       # geodesic((a纬度，a经度），（b纬度，b经度））。米
            dist = haversine(a[i, 1], a[i, 0], b[j, 1], b[j, 0])                 # haversine(a纬度，a经度），（b纬度，b经度）。米
            last_min = np.min([DTWM[i, j], DTWM[i+1, j], DTWM[i, j+1]])
            DTWM[i+1, j+1] = dist + last_min
    return DTWM


# 柯西分布函数值计算
@jit(nopython=True)
def Cauchy(Distance, LofM):
    if LofM > 0:
        fv = 1 / (3.141592653589793 * LofM * (1 + (Distance / LofM) ** 2))
    else:
        fv = 9999999999999
    return fv


# 取损失矩阵最右下角元素
def DTWMtoDist(DTWM):
    (h, l) = DTWM.shape
    return DTWM[h-1, l-1]


# 柯西分布隶属度DTW计算
def DTWMembershipGrade(a, b):

    n, m = len(a), len(b)
    # 展度计算
    LofMd = sqrt(DTWMtoDist(DTWdist(a, b)) / (n * m))  # 距离因素的展度
    LofMc = sqrt(DTWMtoDist(DTWcourse(a[:, 2], b[:, 2])) / (n * m))  # 航向因素的展度
    LofMs = sqrt(DTWMtoDist(DTW(a[:, 3], b[:, 3])) / (n * m))  # 速度因素的展度

    DTWM = np.zeros((n + 1, m + 1))
    for ix in range(n):
        for jx in range(m):
            # distance = geodesic((a[ix, 1], a[ix, 0]), (b[jx, 1], b[jx, 0])).meters  # geodesic((a纬度，a经度），（b纬度，b经度））。米
            distance = haversine(a[ix, 1], a[ix, 0], b[jx, 1], b[jx, 0])  # haversine(a纬度，a经度），（b纬度，b经度）。米
            Gdist = Cauchy(distance, LofMd)
            Gcource = Cauchy(deltaCourse(a[ix, 2], b[jx, 2]), LofMc)
            Gspeed = Cauchy(abs(a[ix, 3]-b[jx, 3]), LofMs)
            # Grade = (Gdist + Gcource + Gspeed)/3  # 总隶属度三个因素的权重，目前为三个因素权重相等
            Grade = Gdist + Gcource + Gspeed
            last_max = np.max([DTWM[ix, jx], DTWM[ix + 1, jx], DTWM[ix, jx + 1]])
            DTWM[ix + 1, jx + 1] = Grade + last_max
    return DTWM


# 构造轨迹数组
def toTrajectory(dataA, dataB):
    # 统计有几艘船的轨迹
    dataAid = dataA['batch'].drop_duplicates().values
    dataBid = dataB['batch'].drop_duplicates().values
    n_ships = np.min([len(dataAid), len(dataBid)])
    dataA_n_ships = len(dataAid)
    dataB_n_ships = len(dataBid)
    trajectoryA = []
    trajectoryB = []
    for i in range(dataA_n_ships):
        tempA = dataA[dataA['batch'] == dataAid[i]].sort_values('time')
        trajectoryA.append(tempA[['lon', 'lat', 'cou', 'vel', 'time']])
    for j in range(dataB_n_ships):
        tempB = dataB[dataB['batch'] == dataBid[j]].sort_values('time')
        trajectoryB.append(tempB[['lon', 'lat', 'cou', 'vel', 'time']])
    return dataAid, dataBid, dataA_n_ships, dataB_n_ships, n_ships, trajectoryA, trajectoryB


# 匹配轨迹
def MatchingTraj(method, dataAid, dataBid, dataA_n_ships, dataB_n_ships, n_ships, trajectoryA, trajectoryB):
    # 计算隶属度矩阵
    GradeArray = np.zeros((dataA_n_ships, dataB_n_ships))  # 隶属度矩阵
    time_sum = 0
    for i in range(dataA_n_ships):
        for j in range(dataB_n_ships):
            # time1 = time()
            print('开始计算9001batch%s与9002batch%s' % (dataAid[i], dataBid[j]))
            trajEndtime = min([trajectoryA[i].values[-1][4], trajectoryB[j].values[-1][4]])
            trajStarttime = max([trajectoryA[i].values[0][4], trajectoryB[j].values[0][4]])
            if (trajEndtime - trajStarttime) >= 120:
                if method == 'CM':
                    trajB = trajectoryB[j][
                        (trajectoryB[j]['time'] >= trajStarttime) & (trajectoryB[j]['time'] <= trajEndtime)].values
                    if len(trajB) > 0:
                        trajA = trajectoryA[i][
                            (trajectoryA[i]['time'] > trajB[0, -1]) & (trajectoryA[i]['time'] < trajB[-1, -1])].values
                    else:
                        Grade = (-9999999999999)
                else:
                    trajA = trajectoryA[i][
                        (trajectoryA[i]['time'] >= trajStarttime) & (trajectoryA[i]['time'] <= trajEndtime)].values
                    trajB = trajectoryB[j][
                        (trajectoryB[j]['time'] >= trajStarttime) & (trajectoryB[j]['time'] <= trajEndtime)].values
                if (len(trajA) > 0) & (len(trajB) > 0):
                    if method == 'DTWCM':
                        MGmatrix = DTWMembershipGrade(trajA, trajB)  # 使用DTW多因素柯西分布隶属度
                        Grade = DTWMtoDist(MGmatrix) / max(len(trajA), len(trajB))
            else:
                Grade = (-9999999999999)
            GradeArray[i, j] = Grade
            print('隶属度为%.8f' % (Grade * 100), '%')
            # time2 = time()
            # print('耗时', time2 - time1, 's')
            # time_sum += (time2 - time1)
    # print('完成隶属度矩阵计算，总耗时', time_sum, '秒')
    # 根据隶属度矩阵规划求解，匹配对应轨迹
    gradeMaxLP = pulp.LpProblem("System_Optimal", sense=pulp.LpMaximize)  # 优化问题gradeMaxLP为使总隶属度最大的0-1整数规划，即为系统最优原则的轨迹匹配
    matrixXX = [[pulp.LpVariable('x%d,%d' % (i, j), cat='Binary') for j in dataBid] for i in dataAid]  # 定义变量xij为是否匹配9001中第i个轨迹和9002中第j个轨迹，xij=0或1，xij组成0-1变量矩阵matrixXX，matrixXX[i,j]=xij
    GradeArray = np.nan_to_num(GradeArray, nan=-9999999999999)
    gradeMaxLP += (np.vdot(GradeArray, matrixXX))  # 目标函数，隶属度矩阵与0-1变量矩阵的点积
    # 约束条件
    gradeMaxLP += (np.sum(matrixXX) <= n_ships)  # 匹配的轨迹对不能超过9001或9002中较少的轨迹数量
    for xj in range(dataB_n_ships):
        gradeMaxLP += (sum([x[xj] for x in matrixXX]) == 1)  # 每个来自9002的轨迹只能匹配一个来自9001的轨迹
    for xi in range(dataA_n_ships):
        gradeMaxLP += (sum(matrixXX[xi]) == 1)  # 每个来自9001的轨迹只能匹配一个来自9002的轨迹或不匹配轨迹
    # 求解
    gradeMaxLP.solve()
    # 输出求解状态
    results = {}
    print('以下为解')
    for yj in range(dataB_n_ships):
        for yi in range(dataA_n_ships):
            if matrixXX[yi][yj].varValue == 1:
                print(matrixXX[yi][yj])
                results.update({str(dataAid[yi]): str(dataBid[yj])})
    return results


def runMatch(method, er, nr, task):
    print(task)
    for e in er:
        for n in nr:
            testdataA = read_csv(
                "D:/pyProject/track-to-track association/testdata/simData/expCTnewGSnew/scene-%d_error_%d_AIS.csv" % (
                n, e),
                header=0, index_col=None)
            testdataB = read_csv(
                "D:/pyProject/track-to-track association/testdata/simData/expCTnewGSnew/scene-%d_error_%d_radar.csv" % (
                n, e),
                header=0, index_col=None)
            dataAid, dataBid, dataA_n_ships, dataB_n_ships, n_ships, trajectoryA, trajectoryB = toTrajectory(testdataA, testdataB)
            try:
                re = MatchingTraj(method, dataAid, dataBid, dataA_n_ships, dataB_n_ships, n_ships, trajectoryA,
                                  trajectoryB)
                with open(
                        'D:/pyProject/track-to-track association/testdata/simData/GS/save%s/scene-%d_error_%d.csv' % (method, n, e), 'w') as f:
                    json.dump(re, f)
                f.close()
            except:
                print('%s场景-%d_error_%d.csv' % (method, n, e), '出错')


if __name__ == '__main__':

    nset = [0, 3, 6, 7, 35, 37, 44, 78, 85, 100]
    p = Pool(5)

    p.apply_async(runMatch, args=('DTWCM', [0, 1, 2], nset, 'task_DTWCM'))
    p.apply_async(runMatch, args=('DTWCM', [3, 4], nset, 'task_DTWCM'))
    p.apply_async(runMatch, args=('DTWCM', [5, 6], nset, 'task_DTWCM'))
    p.apply_async(runMatch, args=('DTWCM', [7, 8], nset, 'task_DTWCM'))
    p.apply_async(runMatch, args=('DTWCM', [9, 10], nset, 'task_DTWCM'))

    p.close()
    p.join()















