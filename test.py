import gym


# env = gym.make('LunarLander-v2')
env = gym.make('InvertedPendulum-v2')
env = gym.wrappers.Monitor(env, './vidoes', force=True)
env.reset()
done = False
while not done:
    env.render()
    ns, r, done, _ = env.step(env.action_space.sample())
