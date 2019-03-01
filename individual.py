import random


class individual:

    PBest = []
    GBest = []
    bestCost= 0
    velocity = []
    pos = []

    w = 0.2
    c1 = 1.49
    c2 = 1.49

    def __init__(self, pos, PBest, velocity):

        self.pos = pos
        self.PBest = PBest
        self.velocity = velocity


    def updateBest (self, cost):

        if (cost<self.bestCost):

            self.bestCost = cost
            self.PBest = self.pos


    def setCost(self, cost):

        self.bestCost = cost


    def update_GBest(self, gbest):

        self.GBest = gbest


    def update_position(self):

        for i in range(0, len(self.velocity)):
            s = self.w * self.velocity[i] + self.c1 * random.uniform(0,1) * (self.PBest[i] - self.pos[i]) + self.c2 * random.uniform(0,1) * (self.GBest[i] - self.pos[i])
            self.pos[i] =  self.pos[i] + s
            self.velocity[i] = s;
