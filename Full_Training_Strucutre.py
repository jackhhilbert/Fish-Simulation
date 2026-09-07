# The shark's name is Mr_Fish
# The Goal of the fish is to lure Mr_Fish to the edge and kill him
import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.distributions
import matplotlib.pyplot as plt

pygame.init()

# Initialize the Game
WIDTH = 1200
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Important Classes
# Fish Code
class Fish:
    def __init__(self, position):
        # Fish Movement Parameters
        self.position = list(position)
        angle = random.uniform(0, 2 * np.pi)
        self.magnitude = 5
        self.alive = True
        self.vector = [self.magnitude * np.cos(angle),
                       self.magnitude * np.sin(angle)]
        self.edged = False
        # Fish Learning Parameters
        self.log_probs= []
        self.reward_scores = []
        self.reward = 0
        self.took_action = False
        self.edge_penalty = False

    def move(self, brain, shark_position, border):
        # Initialize Parameters for Brain
        # Edge Distance has Options because 4 Walls
        edge_dist_x = ((WIDTH - self.position[0]) - (self.position[0] - 0)) / WIDTH
        edge_dist_y = ((HEIGHT - self.position[1]) - (self.position[1] - 0)) / HEIGHT
        shark_dist_x = (self.position[0] - shark_position[0]) / WIDTH
        shark_dist_y = (self.position[1] - shark_position[1]) / HEIGHT

        # Collect a Direction from Fish Brain
        outputs = brain.forward(edge_dist_x, edge_dist_y, shark_dist_x, shark_dist_y)
        distribution_x = torch.distributions.normal.Normal(loc = outputs[0], scale = torch.tensor(0.3))
        distribution_y = torch.distributions.normal.Normal(loc = outputs[1], scale = torch.tensor(0.3))
        sample_x = distribution_x.sample()
        sample_y = distribution_y.sample()
        log_prob_x = distribution_x.log_prob(sample_x)
        log_prob_y = distribution_y.log_prob(sample_y)

        # Normalize the Directions
        magnitude = torch.sqrt(sample_x**2 + sample_y**2)
        magnitude = torch.clamp(magnitude, min=1e-3)
        normalized_directions = [sample_x/ magnitude, sample_y / magnitude]

        # Convert Tensor to Numpy
        for i in range(len(normalized_directions)):
            normalized_directions[i] = normalized_directions[i].detach().numpy()
        self.log_probs.append(log_prob_x + log_prob_y)
        self.took_action = True

        # Change Vector by Direction
        self.vector = [self.magnitude * normalized_directions[0],
                       self.magnitude * normalized_directions[1]]
        self.position[0] += self.vector[0]
        self.position[1] += self.vector[1]

        # The Fish Die if They Get to the Edge
        if border.circle_collides(self.position, 7):
            self.alive = False
            self.edge_penalty = True

    def learn(self, hunted, won, shark_edge_change, shark_following, shark_position):
        self.reward = 0
        # Reward moving the shark toward an edge
        if self.alive and shark_following:
            self.reward += 2 * shark_edge_change
        distance = np.hypot(
            self.position[0] - shark_position[0],
            self.position[1] - shark_position[1]
        )
        target_distance = 120
        distance_error = abs(distance - target_distance)
        self.reward -= 0.01 * distance_error

        if self.edge_penalty:
            self.reward -=20
            self.edge_penalty = False
        # Getting Eaten is Bad
        if hunted:
            self.reward -= 20

        # Killing the shark is the objective
        if won:
            self.reward += 200
        if won and self.alive:
            self.reward += 50

        self.reward_scores.append(self.reward)

    def draw(self):
        pygame.draw.circle(surface = screen,
                            color = (0, 200, 0),
                            center = (int(self.position[0]),
                                      int(self.position[1])),
                            radius = 7)
# Fish Brain Class
class Fish_Brain(nn.Module):
    # The Fish will Share a Brain
    def __init__(self):
        super(Fish_Brain, self).__init__()
        self.inputLayer = nn.Linear(4,16)
        self.outputLayer = nn.Linear(16, 2)
    # Inputs: Distance to Edge, Distance to Shark
        # Inputs: X Distance to Edge, Y Distance to Edge, X Distance to Shark, Y Distance to Shark
    def forward(self, edge_dist_x, edge_dist_y, shark_dist_x, shark_dist_y):
        inputs = torch.tensor([edge_dist_x, edge_dist_y, shark_dist_x, shark_dist_y])
        hidden_layer_1 =  self.inputLayer(inputs)
        hidden_layer_1 = torch.tanh(hidden_layer_1)
        outputs = self.outputLayer(hidden_layer_1)
        outputs = torch.tanh(outputs)
        return outputs

# Shark Code
class Shark:
    def __init__(self):
        self.position = [WIDTH / 2, HEIGHT / 2]
        self.magnitude = 6
        self.alive = True
        self.vector = [0, 0]
    def hunt(self, prey, border):
        if prey is None:
            return
        x_dist = prey.position[0] - self.position[0]
        y_dist = prey.position[1] - self.position[1]
        distance = np.hypot(x_dist, y_dist)
        if distance == 0:
            return
        self.vector = [self.magnitude * x_dist / distance,
                       self.magnitude * y_dist / distance]

        self.position[0] += self.vector[0]
        self.position[1] += self.vector[1]

        # The Shark Dies if it hits the edge
        if border.circle_collides(self.position, 14):
            self.alive = False
    def draw(self):
        pygame.draw.circle(surface = screen,
                           color = (200,0,0),
                           center = (int(self.position[0]),
                                     int(self.position[1])),
                           radius = 14
                           )
class Border:
    def __init__(self):
        self.thickness = 10

        self.top = pygame.Rect(
            0, 0, WIDTH, self.thickness
        )

        self.bottom = pygame.Rect(
            0,
            HEIGHT - self.thickness,
            WIDTH,
            self.thickness
        )

        self.left = pygame.Rect(
            0, 0,
            self.thickness,
            HEIGHT
        )

        self.right = pygame.Rect(
            WIDTH - self.thickness,
            0,
            self.thickness,
            HEIGHT
        )

    def circle_collides(self, position, radius):
        x, y = position

        return (
                x - radius <= self.thickness
                or x + radius >= WIDTH - self.thickness
                or y - radius <= self.thickness
                or y + radius >= HEIGHT - self.thickness
        )

    def draw(self):
        pygame.draw.rect(screen, (200, 200, 200), self.top)
        pygame.draw.rect(screen, (200, 200, 200), self.bottom)
        pygame.draw.rect(screen, (200, 200, 200), self.left)
        pygame.draw.rect(screen, (200, 200, 200), self.right)
collision = Border()
def shark_edge_distance(shark):
    return min(
        shark.position[0],
        WIDTH - shark.position[0],
        shark.position[1],
        HEIGHT - shark.position[1]
    )
def run_instance(fish_brain):
    # Create the 45 Fish
    fishes = []
    for i in range(45):
        fish = Fish(position=[(random.randint(15, WIDTH - 15)),
                              (random.randint(15, HEIGHT - 15))])
        fishes.append(fish)
    # Create 1 Shark
    shark = Shark()
    quit_requested = False
    # Run The Simulation 1 Time
    sim = True
    frame_count = 0
    shark_died = False
    while sim:
        clock.tick(30)
        frame_count += 1
        game_tick = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim = False
                quit_requested = True
        screen.fill((0, 100, 150))
        collision.draw()

        # Fish Initializations and Hunting Loop
        old_shark_edge_distance = shark_edge_distance(shark)
        closest = None
        closest_dist = np.inf
        for fish in fishes:
            if fish.alive and shark.alive:
                fish.move(fish_brain, shark.position, collision)
                fish.draw()
                distance = np.hypot(fish.position[0] - shark.position[0],
                                    fish.position[1] - shark.position[1])
                if distance < closest_dist:
                    closest = fish
                    closest_dist = distance
        # Move Shark to Closest Fish
        if (closest is not None) and shark.alive:
            shark.hunt(closest, collision)
            distance = np.hypot(closest.position[0] - shark.position[0],
                                closest.position[1] - shark.position[1])

        # Loop for Learning
        new_shark_edge_distance = shark_edge_distance(shark)
        shark_edge_change = old_shark_edge_distance - new_shark_edge_distance
        for fish in fishes:
            hunted = False
            won = False
            if fish.took_action:
                if fish.alive and shark.alive:
                    distance = np.hypot(fish.position[0] - shark.position[0],
                                fish.position[1] - shark.position[1])
                    if distance < 10:
                        fish.alive = False
                        hunted = True
                if not shark.alive:
                    won = True
                shark_following = False
                if closest == fish:
                    shark_following = True
                fish.learn(hunted, won, shark_edge_change, shark_following, shark.position)
                fish.took_action = False

        shark.draw()
        pygame.display.flip()
        if closest is None or shark.alive is False:
            sim = False
            break
    if not shark.alive:
        shark_died = True
    return fishes, quit_requested, frame_count, shark_died

# Training Instance for Updating the Brain
def train_instance(fishes, optimizer):
    gamma = 0.99
    all_log_probs = []
    all_returns = []
    total_rewards = []  # un-normalized, one total per fish that acted
    for fish in fishes:
        discounted_returns = []
        if len(fish.log_probs) > 0:
            G = 0
            for r in reversed(fish.reward_scores):
                G = r + gamma * G
                discounted_returns.insert(0, G)
            all_log_probs.extend(fish.log_probs)
            all_returns.extend(discounted_returns)
            total_rewards.append(sum(fish.reward_scores))

    if len(all_log_probs) == 0:
        return 0.0, 0.0

    avg_reward = sum(total_rewards) / len(total_rewards)

    all_returns = torch.tensor(all_returns)
    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 0.00000001)
    all_log_probs = torch.stack(all_log_probs)
    loss = -(all_log_probs * all_returns).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), avg_reward

# Plot to See Improvements
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 7))
fig.subplots_adjust(hspace=1)

runtime_line, = ax1.plot([], [], marker='o', markersize=2.3, linewidth=1)
ax1.set_xlabel("Episode Number")
ax1.set_ylabel("Runtime (frames)")
ax1.set_title("Episode Runtime")

death_rate_line, = ax2.plot([], [], color='red', linewidth=2)
ax2.set_xlabel("Episode Number")
ax2.set_ylabel("Shark Death Rate (%)")
ax2.set_title("Cumulative Shark Death Rate")
ax2.set_ylim(0, 100)

reward_line, = ax3.plot([], [], color='green', marker='o', markersize=2.3, linewidth=1)
ax3.set_xlabel("Episode Number")
ax3.set_ylabel("Avg Reward per Fish")
ax3.set_title("Average Raw Reward per Fish per Episode")

# Brain Parameters and Plotting Info
fish_brain = Fish_Brain()
optimizer = torch.optim.Adam(fish_brain.parameters(), lr = 0.001)
episode_numbers = []
runtimes = []
shark_died_flags = []
shark_death_rates = []

episode_numbers = []
runtimes = []
shark_died_flags = []
shark_death_rates = []
avg_rewards = []  # new

for i in range(5000):
    pygame.display.set_caption("Simulation " + str(i) + " of Mr. Fish Hunting")
    fishes, quit_requested, frame_count, shark_died = run_instance(fish_brain)
    loss, avg_reward = train_instance(fishes, optimizer)  # now unpacks two values

    if quit_requested:
        break
    episode_numbers.append(i)
    runtimes.append(frame_count)
    shark_died_flags.append(shark_died)
    avg_rewards.append(avg_reward)

    number_of_shark_deaths = sum(shark_died_flags)
    total_episodes = len(shark_died_flags)
    shark_death_rate = (number_of_shark_deaths / total_episodes) * 100
    shark_death_rates.append(shark_death_rate)

    print(
        "Episode:", i,
        "| Loss:", round(loss, 4),
        "| Avg Reward:", round(avg_reward, 3),
        "| Runtime:", frame_count,
        "| Shark Died:", shark_died,
        "| Shark Death Rate:", round(shark_death_rate, 2), "%"
    )

    runtime_line.set_data(episode_numbers, runtimes)
    ax1.relim()
    ax1.autoscale_view()

    death_rate_line.set_data(episode_numbers, shark_death_rates)
    ax2.relim()
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, max(10, i + 1))

    reward_line.set_data(episode_numbers, avg_rewards)  # new
    ax3.relim()
    ax3.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

pygame.quit()
plt.ioff()
plt.show()

