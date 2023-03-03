import os as os
import numpy as np
import typing as typ
import random as ran
import colorama as col
import tensorflow as tf
import collections as co
import matplotlib.pyplot as plt
import tensorflow.keras.layers as lay
import tensorflow.keras.models as mod
import tensorflow.keras.losses as los
import tensorflow.keras.optimizers as opt
import tensorflow.keras.activations as act
import tensorflow.keras.initializers as ini

class WORLD:
    def __init__(self,
                 MinT:typ.Union[int, float]=-100,
                 MaxT:typ.Union[int, float]=+100,
                 nEpisode:int=300,
                 mStep:int=70,
                 qLR:float=9e-1,
                 Gamma:float=9e-1,
                 Epsilon0:float=1,
                 Epsilon1:float=2e-2,
                 nDecayEpisode:int=200,
                 nDense:list[int]=[256, 256],
                 Activation:str='elu',
                 Optimizer:str='Adam',
                 Loss:str='huber',
                 mLR:float=3e-3,
                 nEpoch:int=1,
                 TrainOn:int=16,
                 sBatch:int=32,
                 sMemory:int=1024,
                 Verbose:int=0,
                 RandomState:typ.Union[int, None]=None):
        self.MinT = MinT
        self.MaxT = MaxT
        self.nEpisode = nEpisode
        self.mStep = mStep
        self.qLR = qLR
        self.Gamma = Gamma
        self.Epsilon0 = Epsilon0
        self.Epsilon1 = Epsilon1
        self.nDecayEpisode = nDecayEpisode
        self.nDense = nDense
        self.Activation = Activation
        self.Optimizer = Optimizer
        self.Loss = Loss
        self.mLR = mLR
        self.nEpoch = nEpoch
        self.TrainOn = TrainOn
        self.sBatch = sBatch
        self.sMemory = sMemory
        self.Verbose = Verbose
        self.RandomState = RandomState
        self.SetRandomState()
        self.ApplySettings()
    def SetRandomState(self):
        if self.RandomState is not None:
            ran.seed(self.RandomState)
            np.random.seed(self.RandomState)
            tf.random.set_seed(self.RandomState)
            os.environ['PYTHONHASHSEED'] = str(self.RandomState)
    def ApplySettings(self):
        plt.style.use('ggplot')
        self.Code2Actions = {0: 'Down', 1: 'Up'}
        self.Action2Change = {0: -1, 1: +1}
        self.nAction = len(self.Code2Actions)
        self.Episode = -1
        nConstantEpisode = self.nEpisode - self.nDecayEpisode
        epsA = np.linspace(start=self.Epsilon0,
                           stop=self.Epsilon1,
                           num=self.nDecayEpisode)
        epsB = self.Epsilon1 * np.ones(nConstantEpisode)
        self.Epsilons = np.hstack((epsA, epsB))
        self.ActionLog = np.zeros((self.nEpisode, self.mStep))
        self.EpisodeLog = np.zeros(self.nEpisode)
        self.oActivaiton = 'linear'
        States = np.linspace(start=self.MinT,
                             stop=self.MaxT,
                             num=201)
        Rewards = self.State2Reward(States)
        self.MinReward = Rewards.min()
        self.MaxReward = Rewards.max()
        self.MinQ = (1 + self.qLR * self.Gamma) * self.MinReward
        self.MaxQ = (1 + self.qLR * self.Gamma) * self.MaxReward
        self.Memory = co.deque(maxlen=self.sMemory)
        self.Counter = 0
        self.a = 2 / (self.MaxT - self.MinT)
        self.b = (self.MinT + self.MaxT) / (self.MinT - self.MaxT)
        self.Optimizer = self.Optimizer.upper()
        self.Loss = self.Loss.upper()
        self.Activation = self.Activation.upper()
        if self.Optimizer == 'ADAM':
            self.Optimizer = opt.Adam(learning_rate=self.mLR)
        elif self.Optimizer == 'SGD':
            self.Optimizer = opt.SGD(learning_rate=self.mLR)
        elif self.Optimizer == 'RMSPROP':
            self.Optimizer = opt.RMSprop(learning_rate=self.mLR)
        if self.Loss == 'MSE':
            self.Loss = los.MeanSquaredError()
        elif self.Loss == 'MAE':
            self.Loss = los.MeanAbsoluteError()
        elif self.Loss == 'HUBER':
            self.Loss = los.Huber(delta=1)
        if self.Activation == 'RELU':
            self.Activation = act.relu
        elif self.Activation == 'ELU':
            self.Activation = act.elu
        elif self.Activation == 'TANH':
            self.Activation = act.tanh
    def State2Reward(self,
                     State:typ.Union[int, float, np.ndarray]) -> typ.Union[float, np.ndarray]:
        Reward = 1 - 0.0002 * State ** 2
        return Reward
    def PlotState2Reward(self):
        States = np.linspace(start=self.MinT,
                             stop=self.MaxT,
                             num=201)
        Rewards = self.State2Reward(States)
        plt.plot(States,
                 Rewards,
                 ls='-',
                 lw=1.2,
                 c='teal')
        plt.title('Reward-State Plot')
        plt.xlabel('State')
        plt.ylabel('Reward')
        plt.show()
    def PlotEpsilons(self):
        T = np.arange(start=1,
                      stop=self.nEpisode + 1,
                      step=1)
        plt.plot(T,
                 self.Epsilons,
                 ls='-',
                 lw=1.2,
                 c='teal')
        plt.title('Epsilon Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.show()
    def StateScaler(self,
                    State:typ.Union[int, float, np.ndarray]) -> typ.Union[float, np.ndarray]:
        ScaledState = self.a * State + self.b
        return ScaledState
    def CreateModel(self):
        self.Model = mod.Sequential()
        self.Model.add(lay.InputLayer(input_shape=(1, )))
        for n in self.nDense:
            BI = ini.RandomUniform(minval=-1,
                                   maxval=+1,
                                   seed=self.RandomState)
            self.Model.add(lay.Dense(units=n,
                                     activation=self.Activation,
                                     bias_initializer=BI))
        BI = ini.RandomUniform(minval=-0.1,
                               maxval=+0.1,
                               seed=self.RandomState)
        self.Model.add(lay.Dense(units=self.nAction,
                                 activation=self.oActivaiton,
                                 bias_initializer=BI))
    def CompileModel(self):
        self.Model.compile(optimizer=self.Optimizer,
                           loss=self.Loss)
    def ModelSummary(self):
        self.HL()
        print('Model Summary:')
        self.Model.summary()
        self.HL()
    def HL(self, s:str='_', n:int=120):
        print(s * n)
    def PredictQ(self,
                 States:typ.Union[int, float, np.ndarray]) -> np.ndarray:
        ScaledStates = self.StateScaler(States)
        if isinstance(States, np.ndarray):
            Q = self.Model.predict(ScaledStates.reshape(-1, 1),
                                   verbose=self.Verbose)
        else:
            Q = self.Model.predict(np.array([[ScaledStates]]),
                                   verbose=self.Verbose)[0]
        return Q
    def SaveModel(self, Path:str='Model'):
        mod.save_model(self.Model, Path)
        print(col.Fore.MAGENTA + 'Model Saved Successfully.' + col.Fore.RESET)
    def LoadModel(self, Path:str='Model'):
        self.Model = mod.load_model(Path)
        print(col.Fore.MAGENTA + 'Model Loaded Successfully.' + col.Fore.RESET)
    def Save2Memory(self,
                    State:float,
                    Action:int,
                    Reward:float,
                    State2:float):
        self.Memory.append([State, Action, Reward, State2])
        self.Counter += 1
    def TrainModel(self):
        Selecteds = np.random.choice(len(self.Memory),
                                     size=self.sBatch)
        X0 = np.zeros((self.sBatch, 1))
        Y = np.zeros((self.sBatch, self.nAction))
        for i, j in enumerate(Selecteds):
            State, Action, Reward, State2 = self.Memory[j]
            oldOutput1 = self.PredictQ(State)
            oldOutput2 = self.PredictQ(State2)
            oldQ1 = oldOutput1[Action]
            TD = Reward + self.Gamma * oldOutput2.max() - oldQ1
            newQ1 = oldQ1 + self.qLR * TD
            newOutput1 = oldOutput1.copy()
            newOutput1[Action] = newQ1
            newOutput1 = np.clip(newOutput1,
                                 a_min=self.MinQ,
                                 a_max=self.MaxQ)
            X0[i] = State
            Y[i] = newOutput1
        X = self.StateScaler(X0)
        self.Model.fit(X,
                       Y,
                       epochs=self.nEpoch,
                       batch_size=self.sBatch,
                       verbose=self.Verbose)
        print(col.Fore.GREEN + 'Model Trained On Batch.' + col.Fore.RESET)
    def Decide(self, Policy:str) -> int:
        if Policy == 'R':
            return np.random.randint(low=0, high=self.nAction)
        elif Policy == 'G':
            q = self.PredictQ(self.State)
            return np.argmax(q)
        elif Policy == 'EG':
            if np.random.rand() < self.Epsilon:
                return self.Decide('R')
            else:
                return self.Decide('G')
    def ResetState(self):
        self.State = np.random.uniform(low=self.MinT,
                                       high=self.MaxT)
        self.Step = -1
    def NextEpisode(self):
        self.Episode += 1
        self.Epsilon = self.Epsilons[self.Episode]
        self.ResetState()
    def NextStep(self):
        self.Step += 1
    def Do(self, Action:int) -> tuple:
        self.State += self.Action2Change[Action]
        if self.State > self.MaxT:
            self.State = self.MaxT
        elif self.State < self.MinT:
            self.State = self.MinT
        Reward = self.State2Reward(self.State)
        if self.Step == self.mStep - 1:
            Done = True
        else:
            Done = False
        return Reward, self.State, Done
    def PlotModelPrediction(self, Title:str):
        States = np.linspace(start=self.MinT,
                             stop=self.MaxT,
                             num=201)
        Rewards = self.State2Reward(States)
        Predictions = self.PredictQ(States)
        plt.plot(States,
                 Rewards,
                 ls='-',
                 lw=1.4,
                 c='k',
                 label='Reward(State)')
        plt.plot(States,
                 Predictions[:, 0],
                 ls='-',
                 lw=1.2,
                 c='r',
                 label='Q(State, Down)')
        plt.plot(States,
                 Predictions[:, 1],
                 ls='-',
                 lw=1.2,
                 c='b',
                 label='Q(State, Up)')
        plt.title(Title)
        plt.xlabel('State')
        plt.ylabel('Reward / Q')
        plt.legend()
        plt.show()
    def Train(self, Policy:str='EG'):
        for _ in range(self.nEpisode):
            self.NextEpisode()
            if self.Episode % 20 == 0:
                self.SaveModel()
                self.PlotModelPrediction(f'Model Prediction For Q Against Real Values (Episode {self.Episode + 1})')
            print(col.Fore.CYAN + f'Episode: {self.Episode + 1} / {self.nEpisode}' + col.Fore.RESET)
            State = self.State
            while True:
                self.NextStep()
                Action = self.Decide(Policy=Policy)
                Reward, State2, Done = self.Do(Action)
                self.Save2Memory(State, Action, Reward, State2)
                if self.Counter == self.TrainOn:
                    self.TrainModel()
                    self.Counter = 0
                self.ActionLog[self.Episode, self.Step] = Reward
                self.EpisodeLog[self.Episode] += Reward
                State = State2
                if Done:
                    break
            print(col.Fore.YELLOW + f'Reward: {self.EpisodeLog[self.Episode]:.4f}' + col.Fore.RESET)
            self.HL()
    def SMA(self, S:np.ndarray, L:int):
        M = np.convolve(S, np.ones(L) / L, mode='valid')
        return M
    def PlotActionLog(self, L:int=300):
        S = self.ActionLog.reshape(-1)
        M = self.SMA(S, L)
        T = np.arange(start=1,
                      stop=S.size + 1,
                      step=1)
        plt.plot(T,
                 S,
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
        plt.axhline(y=self.MinReward,
                    ls='-',
                    lw=1.2,
                    c='k',
                    label='Min Reward')
        plt.axhline(y=self.MaxReward,
                    ls='-',
                    lw=1.2,
                    c='k',
                    label='Max Reward')
        plt.title('Agent Reward On Each Action')
        plt.xlabel('Action')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()
    def PlotEpisodeLog(self, L:int=30):
        M = self.SMA(self.EpisodeLog, L)
        T = np.arange(start=1,
                      stop=self.EpisodeLog.size + 1,
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
        plt.axhline(y=self.mStep * self.MinReward,
                    ls='-',
                    lw=1.2,
                    c='k',
                    label='Min Reward')
        plt.axhline(y=self.mStep * self.MaxReward,
                    ls='-',
                    lw=1.2, 
                    c='k',
                    label='Max Reward')
        plt.title('Agent Reward On Each Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()
    def Test(self, Policy:str='G', Plot:bool=True):
        self.ResetState()
        print(col.Fore.CYAN + f'Testing Agent:' + col.Fore.RESET)
        States = []
        Rewards = []
        State = self.State
        States.append(State)
        while True:
            self.NextStep()
            Action = self.Decide(Policy=Policy)
            Reward, State2, Done = self.Do(Action)
            States.append(State2)
            Rewards.append(Reward)
            if Done:
                break
        print(col.Fore.YELLOW + f'Reward: {sum(Rewards):.4f}' + col.Fore.RESET)
        self.HL()
        if Plot:
            plt.plot(States,
                     ls='-',
                     lw=1.2,
                     c='teal',
                     label='Temperature')
            plt.axhline(y=self.MinT,
                        ls='-',
                        lw=1.2,
                        c='k',
                        label='Min Temperature')
            plt.axhline(y=self.MaxT,
                        ls='-',
                        lw=1.2,
                        c='k',
                        label='Max Temperature')
            plt.title('Agent State Over Test Episode')
            plt.xlabel('Step')
            plt.ylabel('State')
            plt.legend()
            plt.show()
            T = np.arange(start=1,
                          stop=len(Rewards) + 1,
                          step=1)
            plt.plot(T,
                     Rewards,
                     ls='-',
                     lw=1.2,
                     c='teal',
                     label='Reward')
            plt.axhline(y=self.MinReward,
                        ls='-',
                        lw=1.2,
                        c='k',
                        label='Min Reward')
            plt.axhline(y=self.MaxReward,
                        ls='-',
                        lw=1.2,
                        c='k',
                        label='Max Reward')
            plt.title('Agent Reward Over Test Episode')
            plt.xlabel('Action')
            plt.ylabel('Reward')
            plt.legend()
            plt.show()

World = WORLD()

World.CreateModel()
World.CompileModel()
World.ModelSummary()

World.PlotEpsilons()
World.PlotState2Reward()

World.Train()

World.PlotModelPrediction('Model Prediction For Q Against Real Values (Final)')

World.SaveModel()

World.PlotActionLog()

World.PlotEpisodeLog()


for _ in range(10):
    World.Test()
