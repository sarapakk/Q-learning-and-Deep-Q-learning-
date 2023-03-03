import numpy as np
import colorama as col
from typing import Union
import matplotlib.pyplot as plt

class WORLD:
    def __init__(self,
                 H:int,
                 W:int,
                 nEpisode:int=600,
                 mStep:int=120,
                 LR:float=1e-1,
                 Gamma:float=9.9e-1,
                 eps0:float=9.8e-1,
                 eps1:float=0):
        self.H = H
        self.W = W
        self.nEpisode = nEpisode
        self.mStep = mStep
        self.LR = LR
        self.Gamma = Gamma
        self.eps0 = eps0
        self.eps1 = eps1
        self.ApplySettings()
    def ApplySettings(self):
        self.Transition = {0: [-1, 0], # Up
                           1: [0, +1], # Right
                           2: [+1, 0], # Down
                           3: [0, -1]} # Left
        self.nAction = len(self.Transition)
        self.Rewards = {'Outside': -1, # Trying To Go Outside
                        'Move': -0.1, # Normal Move
                        'Hole': -2, # Falling In Hole
                        'Goal': +100} # Reaching Goal
        self.Type2Code = {'Frozen': 0,
                          'Start': 1,
                          'Hole': 2,
                          'Goal': 3,
                          'Outside': 4}
        self.MinQ = -1
        self.MaxQ = +1
        self.Episode = -1
        self.Epsilons = np.linspace(start=self.eps0,
                                    stop=self.eps1,
                                    num=self.nEpisode)
        self.Map = np.zeros((self.H, self.W), dtype=np.int32)
        self.Q = {}
        self.ActionLog = []
        self.EpisodeLog = np.zeros(self.nEpisode)
    def ShowMap(self):
        plt.imshow(self.Map)
        plt.title('World Map')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.show()
    def AddStart(self, h:int, w:int):
        self.Start = np.array([h, w])
        self.Map[h, w] = self.Type2Code['Start']
    def AddGoal(self, h:int, w:int):
        self.Map[h, w] = self.Type2Code['Goal']
    def AddHoles(self, Holes:np.ndarray):
        for i in Holes:
            self.Map[i[0], i[1]] = self.Type2Code['Hole']
    def ResetState(self):
        self.Position = self.Start
        self.UpdateState()
    def UpdateState(self):
        self.s = ''
        for i in [-1, 0, +1]:
            for j in [-1, 0, +1]:
                h = self.Position[0] + i
                w = self.Position[1] + j
                if h in range(self.H) and w in range(self.W):
                    self.s += str(self.Map[h, w])
                else:
                    self.s += str(self.Type2Code['Outside'])
        if self.s not in self.Q:
            self.Q[self.s] = np.random.uniform(low=self.MinQ,
                                               high=self.MaxQ,
                                               size=self.nAction)
    def Decide(self, Policy:str):
        if Policy == 'R': # Random Policy
            a = np.random.randint(low=0, high=4)
        elif Policy == 'G': # Greedy Policy
            t = self.Q[self.s]
            a = np.argmax(t)
        elif Policy == 'EG': # Epsilon-Greedy Policy
            if np.random.rand() < self.Epsilon:
                a = self.Decide(Policy='R')
            else:
                a = self.Decide(Policy='G')
        return a
    def Move2(self, NewPosition:np.ndarray):
        self.Position = NewPosition
        self.UpdateState()
    def DoAction(self, a:int):
        h, w = self.Position + self.Transition[a]
        done = False
        message = None
        NewPosition = np.array([h, w])
        if h not in range(self.H) or w not in range(self.W):
            r = self.Rewards['Outside']
            NewPosition = self.Position
        else:
            if self.Map[h, w] in [self.Type2Code['Start'], self.Type2Code['Frozen']]:
                r = self.Rewards['Move']
            elif self.Map[h, w] == self.Type2Code['Hole']:
                r = self.Rewards['Hole']
                done = True
                message = col.Fore.RED + 'Failed To Reach Goal.' + col.Fore.RESET
            elif self.Map[h, w] == self.Type2Code['Goal']:
                r = self.Rewards['Goal']
                done = True
                message = col.Fore.GREEN + 'Reached Goal.' + col.Fore.RESET
        self.Move2(NewPosition)
        self.ActionLog.append(r)
        return r, self.s, done, message
    def UpdateQ(self, s:str, a:int, r:float, s2:str):
        TD = r + self.Gamma * self.Q[s2].max() - self.Q[s][a]
        self.Q[s][a] = self.Q[s][a] + self.LR * TD
    def NextEpisode(self):
        self.Episode += 1
        self.Epsilon = self.Epsilons[self.Episode]
    def Train(self, Policy:str='EG'):
        for i in range(self.nEpisode):
            self.NextEpisode()
            print(f'Episode {self.Episode + 1} / {self.nEpisode}')
            self.ResetState()
            s = self.s
            for _ in range(self.mStep):
                a = self.Decide(Policy)
                r, s2, done, message = self.DoAction(a)
                self.UpdateQ(s, a, r, s2)
                self.EpisodeLog[i] += r
                s = s2
                if done:
                    break
            if done:
                print(message)
            else:
                print(col.Fore.BLUE + 'Maximum Steps Reached.' + col.Fore.RESET)
            print(col.Fore.YELLOW + f'Reward: {self.EpisodeLog[i]:.4f}' + col.Fore.RESET)
            self.HL()
    def HL(self, s:str='_', n:int=65):
        print(s * n)
    def PlotActionLog(self, L:int=200):
        M = self.SMA(self.ActionLog, L)
        T = np.arange(start=1,
                      stop=len(self.ActionLog) + 1,
                      step=1)
        plt.plot(T,
                 self.ActionLog,
                 ls='-',
                 lw=1.2,
                 c='teal',
                 label='Reward')
        plt.plot(T[-M.size:],
                 M,
                 ls='-',
                 lw=1.4,
                 c='crimson',
                 label=f'SMA({L})')
        plt.title('Agent Reward On Each Action')
        plt.xlabel('Action')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()
    def PlotEpisodeLog(self, L:int=30):
        M = self.SMA(self.EpisodeLog, L)
        T = np.arange(start=1,
                      stop=World.nEpisode + 1,
                      step=1)
        plt.plot(T,
                 self.EpisodeLog,
                 ls='-',
                 lw=1.2,
                 c='teal',
                 label='Reward')
        plt.plot(T[-M.size:],
                 M,
                 ls='-',
                 lw=1.4,
                 c='crimson',
                 label=f'SMA({L})')
        plt.title('Agent Reward On Each Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()
    def SMA(self, S:Union[np.ndarray, list], L:int):
        M = np.convolve(S, np.ones(L) / L, mode='valid')
        return M
    def Test(self, Policy:str='G', Plot:bool=True):
        self.HL()
        print('Testing Agent:')
        self.ResetState()
        Positions = [self.Position]
        Reward = 0
        for _ in range(self.mStep):
            a = self.Decide(Policy)
            r, _, done, message = self.DoAction(a)
            Positions.append(self.Position)
            Reward += r
            if done:
                break
        if done:
            print(message)
        else:
            print(col.Fore.BLUE + 'Maximum Steps Reached.' + col.Fore.RESET)
        print(col.Fore.YELLOW + f'Reward: {Reward:.4f}' + col.Fore.RESET)
        Positions = np.array(Positions)
        print(f'Path:\n{Positions}')
        if Plot:
            xs = []
            ys = []
            cs = []
            for i in range(self.H):
                for j in range(self.W):
                    xs.append(j)
                    ys.append(self.H - i - 1)
                    cs.append(World.Map[i, j])
            plt.scatter(xs,
                        ys,
                        s=1960,
                        c=cs,
                        marker='s')
            for i in range(len(Positions) - 1):
                x1 = Positions[i][1]
                y1 = self.H - Positions[i][0] - 1
                x2 = Positions[i + 1][1]
                y2 = self.H - Positions[i + 1][0] - 1
                dx = x2 - x1
                dy = y2 - y1
                plt.arrow(x1,
                          y1,
                          dx,
                          dy,
                          lw=2,
                          head_width=0.12,
                          head_length=0.1,
                          length_includes_head=True,
                          color='r')
            plt.title('World Map + Agent Path')
            plt.xlabel('Width')
            plt.ylabel('Height')
            plt.xlim(-0.5, self.W - 0.5)
            plt.ylim(-0.5, self.H - 0.5)
            plt.show()
        self.HL()

np.random.seed(0)
plt.style.use('ggplot')

World = WORLD(6, 8)

World.AddStart(0, 0)
World.AddGoal(5, 7)

Holes = np.array([[0, 2],
                  [1, 3],
                  [1, 6],
                  [2, 5],
                  [2, 6],
                  [3, 2],
                  [3, 3],
                  [3, 4],
                  [3, 6],
                  [4, 2],
                  [5, 0],
                  [5, 4],
                  [5, 6]])

World.AddHoles(Holes)

World.ShowMap()

World.Train()

World.PlotActionLog()

World.PlotEpisodeLog()

World.Test()
