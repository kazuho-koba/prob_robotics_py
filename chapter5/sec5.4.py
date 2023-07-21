#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3
import sys  # noqa
sys.path.append('../scripts/')  # noqa
from robot import *  # noqa
import math
from scipy.stats import multivariate_normal

class EstimationAgent(Agent):
    # 初期化：estimatorは自己状態の推定器（今回の場合はMCL）を示す
    def __init__(self, time_interval, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator = estimator
        self.time_interval = time_interval  # 1タイムステップに相当する時間

        # 1つ前の制御指令値で粒子姿勢を更新するので、そのデータを控えておく
        self.prev_nu = 0.0
        self.prev_omega = 0.0

    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        return self.nu, self.omega

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        # elems.append(ax.text(0, 0, "hoge", fontsize=10))

# 各粒子（パーティクル）を定義するクラス
# （粒子はロボットの分身のようなもの、粒子ごとの尤もらしさ、つまり真の自己位置にどの程度近そうかを評価していくということ？）
class Particle:
    def __init__(self, init_pose):
        self.pose = init_pose

    # 実際に粒子の位置を動かすメソッド
    def motion_update(self, nu, omega, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs()       # 順にnn, no, on, oo（式5.9にあるように、並進したときの並進成分雑音、並進したときの旋回成分雑音・・・と4つの雑音を確率分布からドローする）
        
        # 式5.12に従って並進、旋回それぞれにノイズを加える
        noised_nu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)   
        noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)

        # 姿勢を更新（RobotじゃなくてIdealRobot？ → この2つの差はノイズ、誤差、誘拐の有無で、
        # 今はロボット本体でなくその分身である粒子なら誘拐とかは考えないからIdealRobotにノイズだけ個別に入れるのが適するということか）
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)

# モンテカルロ位置推定（Monte Carlo Localization）を行うクラス
class Mcl:
    # 初期化：粒子の数numとそれぞれの初期姿勢からParticleのオブジェクトを生成（motion_noise_stdsのデフォルト値は5.3_2.pyでシミュレーションにより決定）
    def __init__(self, init_pose, num, motion_noise_stds={"nn":0.23, "no":0.001 ,"on":0.11, "oo":0.05}):
        self.particles = [Particle(init_pose) for i in range(num)]

        v = motion_noise_stds                                           # 粒子の位置に加わる、ガウス分布に従う雑音に関する標準偏差
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])   # 与えられたリストを対角成分に持つ対角行列を作る（この場合は4*4行列）
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)         # 4次元ガウス分布

    # 粒子を動かす処理（粒子クラスのメソッドを呼び出す）
    def motion_update(self, nu, omega, time):
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)
        # print(self.motion_noise_rate_pdf.cov)

    # 粒子を描画する（5.2節時点ではまだ粒子を散らばらさないので1つの矢印がロボの初期位置に配置されているだけに見える）
    def draw(self, ax, elems):
        # 各粒子のx, y座標のリスト
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        # 各粒子の向きθのリスト（単位ベクトルとして表示）
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]
        # 要素として追加
        elems.append(ax.quiver(xs, ys, vxs, vys, color="blue", alpha=0.5))

#############################
# 状態遷移モデルのデバッグ用（デバッグ対象もこのあと書き換わるのでアンコメントしても動かない可能性が大
#############################
# initial_pose = np.array([0, 0, 0]).T
# estimator = Mcl(initial_pose, 100, motion_noise_stds={"nn":0.01, "no":0.02, "on":0.03, "oo":0.04})
# a = EstimationAgent(0.1, 0.2, 10.0/180*math.pi, estimator)
# estimator.motion_update(0.2, 10.0/180*math.pi, 0.1)
# for p in estimator.particles:
#     print(p.pose)

##############################
# 粒子の移動の様子のチェック（デバッグ対象もこのあと書き換わるのでアンコメントしても動かない可能性が大
##############################
def trial(motion_noise_stds):
    time_interval = 0.1
    world = World(30, time_interval)
    initial_pose = np.array([0, 0, 0]).T
    estimator = Mcl(initial_pose, 100, motion_noise_stds)
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    straight = EstimationAgent(time_interval, 0.1, 0, estimator)
    r = Robot(initial_pose, sensor=None, agent=circling, color='red')
    world.append(r)
    world.draw()
trial({"nn":0.23, "no":0.001, "on":0.11, "oo":0.05})

################################
# 粒子ではなく実際にロボットを100台生成してチェック
################################
time_interval = 0.1
world = World(30, time_interval, debug=False)
for i in range(100):
    r = Robot(np.array([0,0,0]).T, sensor=None, agent=Agent(0.2, 10.0/180*math.pi), color="gray")
    world.append(r)
world.draw()

# # 環境オブジェクトを生成（Sim時間30秒、1ステップ0.1秒）
# world = World(30, 0.1)

# # 環境中にランドマークが3つあるマップオブジェクトを追加
# m = Map()
# for ln in [(-4, 2), (2, -3), (3, 3)]:
#     m.append_landmark(Landmark(*ln))
# world.append(m)

# # ロボットを配備する
# initial_pose = np.array([2, 2, math.pi/6]).T                    # 各粒子の初期姿勢（全部共通？）
# estimator = Mcl(initial_pose, 100)                              # パーティクルフィルタを作る。粒子は100個
# circling = EstimationAgent(0.2, 10.0/180*math.pi, estimator)    # MCLを推定器estimatorとして持つAgentモデルを生成
# r = Robot(initial_pose, sensor=Camera(m), agent=circling)       # Agentを搭載したロボットを生成
# world.append(r)

# # アニメーションを再生
# world.draw()
