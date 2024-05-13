# Specific to the KitchenPartil-V0
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
from d4rl.kitchen.kitchen_envs import KitchenMicrowaveKettleLightSliderV0
from dm_control.mujoco import engine
class KitchenPartialVisualEnviroment(KitchenMicrowaveKettleLightSliderV0):
    # TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'slide cabinet']
    def __init__(self):
        super().__init__()

    def reset_model(self, state):
        reset_pos = state[:30]
        reset_vel = state[30:]
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        self.goal = self._get_task_goal()  #sample a new goal on reset
        return self._get_obs()
    
    def render(self, mode='human',width=128, height=128):
        if mode =='rgb_array':
            camera = engine.MovableCamera(self.sim, 128, 128)
            camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            img = camera.render()
            return img
        else:
            super(KitchenTaskRelaxV1, self).render()

def main():
    import d4rl
    import gym
    from environments.wrappers import KitchenWrapper, VisualObservationWrapper, VIPFeatureExtractorWrapper

    from matplotlib import pyplot as plt

    print("-----------------------------------1")
    env = gym.make("kitchen-partial-v0")
    env = KitchenWrapper(env)
    env = VisualObservationWrapper(env)
    env = VIPFeatureExtractorWrapper(env)
    print("-----------------------------------2")
    
    dataset = d4rl.qlearning_dataset(env)
    obs = dataset["observations"]
    print(obs.shape)
    print("-----------------------------------3")
    
    renderEnv = KitchenPartialVisualEnviroment()
    renderEnv.reset_model(state=obs[0])
    print("-----------------------------------4")
    first = renderEnv.render(mode='rgb_array', width=128, height=128)
    print(first.shape)
    plt.imshow(first)
    plt.show()

    print("-----------------------------------5")

    print(obs[10000] - obs[0])

    renderEnv.reset_model(state=obs[10000])
    second = renderEnv.render(mode='rgb_array', width=128, height=128)
    print(second.shape)
    print(second - first)
    plt.imshow(second)
    plt.show()

    print("-----------------------------------6")

    for i in range(10):
        import random
        renderEnv.reset_model(state=obs[random.randint(0, len(obs))])
        img = renderEnv.render(mode='rgb_array', width=128, height=128)
        plt.imshow(img)
        print(i)
        plt.show()

    print("-----------------------------------7")

    # renderEnv.reset_model(state=obs[0])
    # transformedEnv = VIPFeatureExtractorWrapper(renderEnv)
    # print(transformedEnv.observation_space)
    # transformedObs = transformedEnv.observation(renderEnv._get_obs())
    # print(transformedObs)



    

if __name__ == "__main__":
    main()