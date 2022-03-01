#!/usr/bin/env python3

# system imports
import numpy
import time
from functools import reduce

# local file imports
import qlearn as ql
from env.MjEnv import MjEnv

if __name__ == '__main__':


    # create the Gym environment
    env = MjEnv()


    # # Set the logging system
    # outdir = "/home/luke/mymujoco/rl" + '/training_results'
    # env = gym.wrappers.Monitor(env, outdir, force=True)

    last_time_steps = numpy.ndarray(0)

    # define key learning parameters
    Alpha = 1.0               # learning rate
    Epsilon = 0.1
    Gamma = 0.1               # discount factor
    epsilon_discount = 0.9
    nepisodes = 50

    # Initialises the algorithm that we are going to use for learning
    qlearn = ql.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):

        print("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        env.render()

        # for each episode, we test the robot for nsteps
        for i in range(env.max_episode_steps):
            
            print("############### Start Step=>" + str(i))

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            print("Next action is:", action)

            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            print("Reward is", reward)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            print("# state we were=>" + str(state))
            print("# action that we took=>" + str(action))
            print("# reward that action gave=>" + str(reward))
            print("# episode cumulated_reward=>" + str(cumulated_reward))
            print("# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                print("NOT DONE")
                state = nextState
            else:
                print("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            print("############### END Step=>" + str(i))
            # input("Next Step...PRESS KEY")

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    ql.save(qlearn)
    test = ql.load()

    exit()
    
    print(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # print("Best 100 score: {:0.2f}".format(
    #     reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:]))
    # )

    env.close()
