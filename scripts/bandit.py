import matplotlib.pyplot as plt
import numpy as np

class Bandit:
    def __init__(self,e=0.0,c=0.0):
        self.est_value=np.zeros(5)
        self.num_selected=np.ones(5)
        self.func_list=[self.gauss_arm, self.coin_arm, self.poisson_arm, self.exp_arm, self.crazy_arm]
        self.epsilon=e
        self.confidence_level=c
        self.plot()

    def gauss_arm(self):
        return np.random.normal(2,1)
    
    def coin_arm(self):
        x=np.random.randint(0,2)
        if x==0:
            return 5
        else:
            return -6
        
    def poisson_arm(self):
        return np.random.poisson(2)
    
    def exp_arm(self):
        return np.random.exponential(3)
    
    def crazy_arm(self):
        x=np.random.randint(0,4)
        if x==0:
            return self.gauss_arm()
        elif x==1:
            return self.coin_arm()
        elif x==2:
            return self.poisson_arm()
        else:
            return self.exp_arm()
        
    def episode_iter(self,ind):
        cum_reward=0
        num_iters=10
        for _ in range(num_iters):
            reward=self.func_list[ind]()
            cum_reward+=reward
        
        return float(cum_reward)/num_iters
        
    def plot(self):
        l=100
        m=2000
        final_reward=np.zeros(m)
        x=np.arange(m)
        c=self.confidence_level
        e=self.epsilon
        for i in range(l):
            reward_array=[]
            for j in range(m):
                val= np.array([np.sqrt(np.log(np.sum(self.num_selected))/self.num_selected[i]) for i in range(5)])
                ind=np.argmax(self.est_value+c*val)
                ind=np.random.choice(a=[ind,0,1,2,3,4], p=[1-e, e/5, e/5, e/5, e/5, e/5])
                self.num_selected[ind]+=1
                episode_rew=self.episode_iter(ind)
                reward_array.append(episode_rew)
                self.est_value[ind]+= (1.0/self.num_selected[ind]) * (episode_rew-self.est_value[ind])
            final_reward=final_reward+reward_array
            
        final_reward=final_reward/l

        # plt.figure()
        # plt.bar(['gauss','coin', 'poisson', 'exp', 'crazy'], self.est_value)
        # plt.title("Estimated Values")

        # plt.figure()
        plt.plot(x,final_reward, label=f'e={self.epsilon}, c={self.confidence_level}')
        # plt.set(f'e={self.epsilon}, c={self.confidence_level}')
        plt.title(f'Reward plot')
        plt.xlabel('Episode')
        plt.ylabel('Reward at the end of episode')
        plt.legend()
        return


if __name__=='__main__':
    bandit0=Bandit()
    # bandit1=Bandit(c=2)
    bandit2=Bandit(e=0.01)
    bandit3=Bandit(e=0.1)
    plt.show()
            


    
