import random
import torch
import yaml
import torch.nn.functional as my_neural_network
from torch import nn, optim
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time

#Reference[1]
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Handling hardcoding, taking from YAML file
no_of_episodes = config['training']['total_episodes']
max_no_of_steps_per_episode = config['training']['count_per_episode']
set_epsilon = config['epsilon']['start']
epsilon_decay = config['epsilon']['decay']
epsilon_min = config['epsilon']['end']
show_visual = config['training']['UI']
enable_learning = config['training']['agent_learn']
replay_size = config['training']['replay_buffer_size']
t_step_size = config['training']['step_size']
chunk_size = config['training']['batch_size']
mini_batch = config['training']['minbatch_size']

total_states = config['network']['state_size']
total_actions = config['network']['action_size']
nerual_layer = config['network']['neural_layer']

folder_name = config['training']['folder_name']
model = config['training']['file_name']
epsilon_file = config['training']['epilon_file']
score_file = config['training']['score_card']


epsilon_start = config['epsilon']['start']
epsilon_end = config['epsilon']['end']
epsilon_decay = config['epsilon']['decay']

block_width = config['snake_world']['block_width']

discount_factor = config['learning']['gamma']
interpolation_parameter = config['learning']['interpolation_parameter']
update_rate = config['learning']['interpolation_parameter']
# print(nerual_layer)


class DQN(nn.Module): #neural network class
    def __init__(self, total_states, total_actions):
        super(DQN, self).__init__()
        self.f_layer_1 = nn.Linear(total_states, nerual_layer)
        self.f_layer_2 = nn.Linear(nerual_layer, total_actions)
    
    def forward(self, total_states):
        x = self.f_layer_1(total_states)
        x = my_neural_network.relu(x) #linear conversion
        x = self.f_layer_2(x)
        return x


class Agent: #agent class
    def __init__(self, total_states, total_actions):
        # self.state_size = total_states
        # self.action_size = total_actions
        self.processing_selection = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.processing_selection)
        self.local_network = DQN(total_states, total_actions).to(self.processing_selection)
        self.target_network = DQN(total_states, total_actions).to(self.processing_selection)
        self.adam_optimizer = optim.Adam(self.local_network.parameters(), lr=float(update_rate))
        self.replay_memory = []
        self.counter = 0
        self.epsilon = 0
        self.max_score = 0


    def get_state(self, game): #NO TOUCHING
        #snake head position [x,y]
        snake_head = [game.snake.x[0], game.snake.y[0]]
        snake_direction = ["left", "right", "up", "down"]
        current_direction = []

        #coordinates of snake 6 points
        N, S, E, W, NE, SE, SW, NW  = [
            #North
            game.is_danger([snake_head[0], (snake_head[1]-block_width)]),
            
            #South
            game.is_danger([snake_head[0], (snake_head[1]+block_width)]),
            
            #East
            game.is_danger([snake_head[0]+block_width, snake_head[1]]),

            #West
            game.is_danger([snake_head[0]-block_width, snake_head[1]]),

            #North East
            game.is_danger([snake_head[0]+block_width, (snake_head[1]-block_width)]),

            #South East
            game.is_danger([snake_head[0]+block_width, (snake_head[1]+block_width)]),

            #South West
            game.is_danger([snake_head[0]-block_width, (snake_head[1]+block_width)]),

            #North West
            game.is_danger([snake_head[0]-block_width, (snake_head[1]-block_width)])
            ]
        
        #Current snake direction 4 points [0,1,0,0]
        for i in snake_direction:
            current_direction.append(game.snake.direction == i)
            # if(game.snake.direction == i):
            #     current_direction.append(True)
            # else:
            #     current_direction.append(False)
        
        #Food direction 4 points [0,1,1,0]
        apple = [
            game.apple.x < snake_head[0],
            game.apple.x > snake_head[0],
            game.apple.y < snake_head[1],
            game.apple.y < snake_head[1]
        ]

        # extending the list (8+4+4)
        eight_direction = [N, S, E, W, NE, SE, SW, NW]
        state = eight_direction + current_direction + apple
        return np.array(state, dtype=int) #Works like charm

    def select_action(self, state_representation, epsilon):
        # Please Note that, I used epsilon value in here, ideally we should be renaming it has temperature, but both follows same ideology for decaying
        #https://www.kaggle.com/code/yashsahu/deep-reinforcement-learning-part-4
        state_representation = torch.from_numpy(state_representation).float().unsqueeze(0).to(self.processing_selection)

        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state_representation)

        self.local_network.train()
        #Boltzmann Softmax action selection
        probabilities = my_neural_network.softmax(action_values / epsilon, dim=1).to(self.processing_selection).numpy()[0]
        action = np.random.choice(len(probabilities), p=probabilities)
        #Traditional Methodology from GITHUB reference 2
        # action = 0
        # if random.random() > epsilon:
        #     action = torch.argmax(action_values).item()
        # else:
        #     action = random.randint(0, 3)
        move = [0, 0, 0, 0]
        move[action] = 1 #Fliping with 1s
        return move, action        

    def memory_store(self, old_state, action, reward, next_state, done): #DON'T TOUCH
        self.replay_memory.append((old_state, action, reward, next_state, done))
        
        #flush old memory, if size exceeds
        if (len(self.replay_memory)>replay_size):
            del self.replay_memory[0]

        #learn every 4 step
        self.learn_from_experience(t_step_size, chunk_size)
        

    def learn_from_experience(self, t_step, chunk_size): #DON'T TOUCH
        self.counter = self.counter+1
        if (self.counter % t_step == 0 and len(self.replay_memory)>mini_batch):
            experiences = self.sample(mini_batch)
            self.learn(experiences)

    def learn(self, experiences):
        # Reference 2
        states, actions, rewards, next_states, dones = experiences
        next_q_targets = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_network(states).gather(1, actions)
        loss = my_neural_network.mse_loss(q_expected, q_targets)
        self.adam_optimizer.zero_grad()
        loss.backward()
        self.adam_optimizer.step()
        self.soft_update(self.local_network, self.target_network)

    def soft_update(self, local_network, target_network):
        # Reference 2
        for local_params, target_params in zip(local_network.parameters(), target_network.parameters()):
            target_params.data.copy_(
                interpolation_parameter * local_params + (1.0 - interpolation_parameter) * target_params
            )

    def sample(self, chunk_size): 
        # Reference 2
        experiences = random.sample(self.replay_memory, k=chunk_size)
        # [(state, action, reward, next_state, done)]
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.processing_selection)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.processing_selection)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.processing_selection)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.processing_selection)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.processing_selection)

        return states, actions, rewards, next_states, dones
    
    def check_gpu_cpu(self): #DONT TOUCH
        if torch.cuda.is_available():
            print("GPU available")
            return "cuda:0"
        else:
            print("Processing with CPU power")
            return "cpu"


    def check_file_creation_update_model(self, epsilon, score, flag):#DON'T TOUCH
        #Checks for model file and epsilon or else create it
        model_file_location = os.path.join(folder_name, model)
        epsilon_file_location = os.path.join(folder_name, epsilon_file)
        score_file_location = os.path.join(folder_name, score_file)

        if (flag == False):
            if ((os.path.exists(folder_name)) and 
                (os.path.exists(model_file_location)) and 
                (os.path.exists(epsilon_file_location)) and 
                (os.path.exists(score_file_location))):
                #Load previous model
                self.local_network.load_state_dict(torch.load(model_file_location))
                #Load previous epsilon
                with open(epsilon_file_location, 'r') as file:
                    resume_epsilon = float(file.readline().strip()) #convert to float to read
                    agent.epsilon = resume_epsilon
                with open(score_file_location, 'r') as file:
                    max_score = float(file.readline().strip()) #convert to float to read

                print("Files loaded and Resuming...")
                return resume_epsilon, max_score, False
            else:
                if(not os.path.exists(folder_name)):
                    print("Created folder")
                    os.mkdir(folder_name)
                    return set_epsilon, 0, False
        else:
            with open(epsilon_file_location, 'w') as file:
                file.write(str(epsilon))
            with open(score_file_location, 'w') as file:
                file.write(str(score))
            torch.save(self.local_network.state_dict(), model_file_location)


    def plot_values(self, epsilon, rewards, score, episode_number, training_time):

        plt.close('all')



        print("epsilon: ", epsilon, "rewards: ", rewards, "score: ", score)


        plt.bar(episode_number, score, label="Score", color="blue")
        xticks = np.arange(0, len(episode_number), 1000)
        plt.xticks(xticks)
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.title("Score vs Episodes")
        plt.legend()
        plt.show(block=False)

        plt.plot(episode_number, training_time, label="Training Time", color="red")
        xticks = np.arange(0, len(episode_number), 1000)
        plt.xticks(xticks)
        plt.xlabel("Episodes")
        plt.ylabel("Training Time")
        plt.title("Episode vs Training Time")
        plt.legend()
        plt.show(block=False)

        plt.plot(episode_number, epsilons, label="Temperature", color="green")
        xticks = np.arange(0, len(episode_number), 1000)
        plt.xticks(xticks)
        plt.xlabel("Episodes")
        plt.ylabel("Temperature")
        plt.title("Episode vs Temperature")
        plt.legend()
        plt.show(block=False)



        #plotting

        # plt.plot(epsilon)
        # plt.plot(rewards)
        #plt.plot(score)
        #plt.plot(episode_number)


max_score = 0

if __name__ == "__main__":

    #Training flag
    if (show_visual):
        from game import Game
    else:
        from game_no_ui import Game

    game = Game()
    agent = Agent(total_states=total_states, total_actions=total_actions)

    episode_number = []
    scores = []
    training_times = []
    epsilons = []

    
    #check for the model file and folder
    resume_epsilon, max_score, _ = agent.check_file_creation_update_model(epsilon= 0, score=max_score, flag=False)
    epsilon = resume_epsilon

    for i in range (no_of_episodes):
        start_time = time.time()
        #Reset environment
        game.reset()
        rewards = []
        score = max_score
        
        for j in range (max_no_of_steps_per_episode):
            #old state
            old_state = agent.get_state(game)
            
            #action
            control_command, action = agent.select_action(old_state, epsilon)

            #Trigger movement
            reward, done, score = game.run(control_command)
            
            #next state
            next_state = agent.get_state(game)

            #Learn
            agent.memory_store(action= action, done=done, next_state= next_state, old_state=old_state, reward=reward)
            rewards.append(reward)

            if done:
                #Call Graph plot to visualize
                # agent.plot_values(rewards=rewards, epsilon=epsilon ,score=max_score)
                break
        end_time = time.time()
        episode_time = end_time - start_time
        training_times.append(episode_time)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        rewards = sum(rewards)
        max_score = max(agent.max_score, score)

        episode_number.append(i+1)
        scores.append(score)
        epsilons.append(epsilon)
        agent.plot_values(episode_number=episode_number, score=scores, rewards=rewards, epsilon=epsilons, training_time=training_times)

        
        agent.check_file_creation_update_model(epsilon=epsilon, score= max_score, flag=True)
        print("Episode number: ", i)
        # print("Episode number: ", i, "    Current score: ", max_score) # To visualize during NO UI training
        
        



