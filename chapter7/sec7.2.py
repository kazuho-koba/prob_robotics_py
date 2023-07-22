from mcl import *
from scipy.stats import chi2

class GlobalMcl(Mcl):
    def __init__(self, envmap, num, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.55, direction_dev=0.13):
        super().__init__(envmap, np.array([0,0,0]).T, num, motion_noise_stds, distance_dev_rate, direction_dev)
        
        # 各粒子をランダムに初期化する
        for p in self.particles:
            p.pose = np.array([np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0), np.random.uniform(-math.pi, math.pi)]).T



# Global_Mclを試すコード
def kldmcl_global_trial(animation):
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation)

    # 地図生成、ランドマーク追加
    m = Map()
    for ln in [(-4, 2), (2,-3), (3, 3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボットの生成
    initial_pose = np.array([np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0), np.random.uniform(-math.pi, math.pi)]).T
    pf = GlobalMcl(m, 1000)   
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, pf)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color='red')
    world.append(r)

    # アニメーション実行
    world.draw()

    # 姿勢（真値）と姿勢（推定値）を返す
    return (r.pose, pf.pose)

# Global_Mclを試行的に1000回実行する
if __name__=='__main__':
    ok = 0
    for i in range(1000):
        actual, estm = kldmcl_global_trial(True)
        diff = math.sqrt((actual[0]-estm[0])**2 + (actual[1] - estm[1])**2)
        print(i, " 真値： ", actual, " 推定値： ", estm, " 誤差： ", diff)
        if diff <= 1.0:
            ok += 1

    print(ok)