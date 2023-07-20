#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

from scipy.stats import norm, multivariate_normal
import seaborn as sns
import random
import math
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import pandas as pd
import matplotlib
# matplotlib.use('nbagg')


class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []  # register objects for robots and other stuff here
        self.debug = debug  # param for debugging
        self.time_span = time_span
        self.time_interval = time_interval

    def append(self, obj):  # func to register objects
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(4, 4))  # prepare fig of 4*4 inch
        ax = fig.add_subplot(111)  # prepare subplot
        ax.set_aspect('equal')  # set aspect ratio same as coordinate
        ax.set_xlim(-5, 5)  # draw x-axis between -5m*5m
        ax.set_ylim(-5, 5)  # draw y-axis between -5m*5m
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)

        elems = []

        # デバッグモードの場合
        if self.debug:
            for i in range(1000):
                self.one_step(i, elems, ax)  # stop animation when debugging
        # そうでない通常モードの場合
        else:
            # アニメーションの描画
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(
                elems, ax), frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            plt.show()

    # 世界の時刻を1タイムステップ進める関数
    def one_step(self, i, elems, ax):
        # 二重描画を防ぐため1タイムステップ前の描画をすべてクリアする
        while elems:
            elems.pop().remove()    # elemsの要素を削除

        # 時刻を表示する
        elems.append(ax.text(-4.5, 4.5, "t= "+str(i), fontsize=10))

        # オブジェクトそれぞれについて描画を行う
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"):
                obj.one_step(self.time_interval)
        pass


class IdealRobot:
    def __init__(self, pose, agent=None, color="black"):
        self.pose = pose    # set default posture by the argument
        self.r = 0.2        # robot radius in drawing
        self.color = color  # set default drawing color by the argument
        self.agent = agent  # ロボットがエージェントによって操縦されているイメージ
        self.poses = [pose]

    # ロボットの描画を行うメソッド
    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x+self.r*math.cos(theta)  # x coordinate value of robot's tip
        yn = y+self.r*math.sin(theta)  # y coordinate value of robot's tip

        # ロボットの向きを示す線分を描画しelemsに追加する（ax.plotはリストを返してくるのでリスト同士の足し算としている）
        elems += ax.plot([x, xn], [y, yn], color=self.color)

        # ロボット自身を示す円を描画しelemsに追加する（こちらはobjectなのでappendで足す）
        c = patches.Circle(xy=(x, y), radius=self.r,
                           fill=False, color=self.color)  # make a circle indicats robot's body
        elems.append(ax.add_patch(c))   # register the circle generated above to subplot
        
        # ロボットの軌道を描画する
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1]
                         for e in self.poses], linewidth=0.5, color="black")

    # ロボットの1ステップの間での状態遷移を示す関数。オブジェクトを作らなくてもこのメソッドを実行できるよう冒頭に@classmethodとつけている。
    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        # 以下、少し複雑だが対向二輪（差動二輪）ロボットの並進速度nuと旋回速度omegaによる運動モデル
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu*math.cos(t0), nu*math.sin(t0), omega])*time
        else:
            return pose + np.array([nu/omega*(math.sin(t0+omega*time)-math.sin(t0)), nu/omega*(-math.cos(t0+omega*time)+math.cos(t0)), omega*time])

    # ロボットに搭載されたエージェントの意思決定にしたがって並進、旋回速度を得て、それを元に1タイムステップ分の運動を決める
    def one_step(self, time_interval):
        if not self.agent:
            return
        nu, omega = self.agent.decision()
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega


world = World(20, 1)
# world.draw()
straight = Agent(0.2, 0.0)                  # go straight forward
circling = Agent(0.2, 10.0/180*math.pi)     # 0.2[m/s], 10[deg/sec]
robot1 = IdealRobot(np.array([2, 3, math.pi/6]).T,
                    straight)  # make robot's instance
robot2 = IdealRobot(np.array([-2, -1, math.pi/5*6]).T, circling, "red")
robot3 = IdealRobot(np.array([0, 0, 0]).T, color="blue")  # robot without agent
world.append(robot1)
world.append(robot2)
world.append(robot3)
world.draw()

# print(IdealRobot.state_transition(0.1, 0.0, 1.0, np.array([0, 0, 0]).T))
