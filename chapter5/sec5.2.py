#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3
import sys  # noqa
sys.path.append('../scripts/')  # noqa
from robot import *  # noqa
import math


class EstimationAgent(Agent):
    # 初期化：estimatorは自己状態の推定器（今回の場合はMCL）を示す
    def __init__(self, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator = estimator

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        # elems.append(ax.text(0, 0, "hoge", fontsize=10))

# 各粒子（パーティクル）を定義するクラス
# （粒子はロボットの分身のようなもの、粒子ごとの尤もらしさ、つまり真の自己位置にどの程度近そうかを評価していくということ？）
class Particle:
    def __init__(self, init_pose):
        self.pose = init_pose

# モンテカルロ位置推定（Monte Carlo Localization）を行うクラス
class Mcl:
    # 初期化：粒子の数numとそれぞれの初期姿勢からParticleのオブジェクトを生成
    def __init__(self, init_pose, num):
        self.particles = [Particle(init_pose) for i in range(num)]

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


# 環境オブジェクトを生成（Sim時間30秒、1ステップ0.1秒）
world = World(30, 0.1)

# 環境中にランドマークが3つあるマップオブジェクトを追加
m = Map()
for ln in [(-4, 2), (2, -3), (3, 3)]:
    m.append_landmark(Landmark(*ln))
world.append(m)

# ロボットを配備する
initial_pose = np.array([2, 2, math.pi/6]).T                    # 各粒子の初期姿勢（全部共通？）
estimator = Mcl(initial_pose, 100)                              # パーティクルフィルタを作る。粒子は100個
circling = EstimationAgent(0.2, 10.0/180*math.pi, estimator)    # MCLを推定器estimatorとして持つAgentモデルを生成
r = Robot(initial_pose, sensor=Camera(m), agent=circling)       # Agentを搭載したロボットを生成
world.append(r)

# アニメーションを再生
world.draw()
