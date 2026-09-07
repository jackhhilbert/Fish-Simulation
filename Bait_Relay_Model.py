# The shark's name is Mr_Fish
# The Goal of the fish is to lure Mr_Fish to the edge and kill him
import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.distributions
import matplotlib.pyplot as plt
from sympy.codegen.ast import none

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
        self.target_direction = np.array([0,0])
        self.relay_alignment = 0.0

    def move(self, brain, shark_position, border, fishes):
        # Initialize Parameters for Brain
        # Edge Distance has Options because 4 Walls
        left_edge_distance = (self.position[0]) / WIDTH
        right_edge_distance = (WIDTH - self.position[0] )/ WIDTH
        top_edge_distance = (self.position[1]) / HEIGHT
        bottom_edge_distance = (HEIGHT - self.position[1]) / HEIGHT
        nearest_edge_distance = min(
            self.position[0],
            WIDTH - self.position[0],
            self.position[1],
            HEIGHT - self.position[1]
        ) / (min(WIDTH, HEIGHT) / 2)

        shark_dx = (shark_position[0] - self.position[0])
        shark_dy = (shark_position[1] - self.position[1])
        shark_distance_pixels = np.hypot(shark_dx, shark_dy)
        shark_distance = shark_distance_pixels / np.hypot(WIDTH, HEIGHT)
        shark_direction_x = 0.0
        shark_direction_y = 0.0
        if shark_distance_pixels > 0:
            shark_direction_x = (shark_dx /shark_distance_pixels)
            shark_direction_y = (shark_dy /shark_distance_pixels)

        # Fish Inputs
        nearby_fish_num = 0
        nearby_fish = []
        for f in fishes:
            if f is self:
                continue
            if not f.alive:
                continue
            distance = np.hypot(
                f.position[0] - self.position[0],
                f.position[1] - self.position[1]
            )
            if distance < 200:
                nearby_fish_num += 1
                nearby_fish.append(f)
        # Scaled from 0 to 1
        nearby_fish_density = nearby_fish_num / 44.0
        # Fish Relay Team Dynamics
        shark_left = shark_position[0]
        shark_right = WIDTH - shark_position[0]
        shark_top = shark_position[1]
        shark_bottom = HEIGHT - shark_position[1]

        nearest_shark_edge = min(
            shark_left,
            shark_right,
            shark_top,
            shark_bottom
        )
        shark_target_edge_distance = nearest_shark_edge / (min(WIDTH, HEIGHT) / 2)
        target_direction = np.array([0,0])
        if nearest_shark_edge == shark_left:
            target_direction = np.array([-1.0, 0.0])
        if nearest_shark_edge == shark_right:
            target_direction = np.array([1.0, 0.0])
        if nearest_shark_edge == shark_top:
            target_direction = np.array([0.0, -1.0])
        if nearest_shark_edge == shark_bottom:
            target_direction = np.array([0.0, 1.0])
        self.target_direction = target_direction
        fish_to_shark_dx = self.position[0] - shark_position[0]
        fish_to_shark_dy = self.position[1] - shark_position[1]

        fish_to_shark_distance = np.hypot(
            fish_to_shark_dx,
            fish_to_shark_dy
        )
        relay_distance = fish_to_shark_distance / np.hypot(WIDTH, HEIGHT)
        relay_alignment = 0
        if fish_to_shark_distance > 1e-3:
            relay_alignment = np.dot(np.array([
                    fish_to_shark_dx / fish_to_shark_distance,
                    fish_to_shark_dy / fish_to_shark_distance
                ]),
                target_direction
            )
        self.relay_alignment = relay_alignment
        fish_ahead_count = 0
        for f in fishes:
            if f is self:
                continue
            if not f.alive:
                continue
            other_vector = np.array([
                f.position[0] - self.position[0],
                f.position[1] - self.position[1]
            ])
            other_distance = np.linalg.norm(other_vector)
            if other_distance < 1e-3:
                continue
            # Normalize vector
            other_direction = (other_vector /other_distance)
            alignment = np.dot(other_direction, target_direction)
            if alignment > 0.5 and other_distance < 200:
                fish_ahead_count += 1
        # Normalize
        fish_ahead_normalized = fish_ahead_count / 44.0
        closest_teammate = None
        closest_teammate_distance = np.inf
        for f in fishes:
            if f is self:
                continue
            if not f.alive:
                continue
            distance = np.hypot(
                f.position[0] - self.position[0],
                f.position[1] - self.position[1]
            )
            if distance < closest_teammate_distance:
                closest_teammate_distance = distance
                closest_teammate = f
        if closest_teammate is None:
            closest_teammate_distance = 1.0
        else:
            closest_teammate_distance /= np.hypot(WIDTH, HEIGHT)

        # Check for Teammates that will Lead Shark To Edge
        teammates_between_me_and_edge = 0
        my_edge_distance = 0
        if nearest_shark_edge == shark_left:
            my_edge_distance = self.position[0]
        if nearest_shark_edge == shark_right:
            my_edge_distance = WIDTH - self.position[0]
        if nearest_shark_edge == shark_top:
            my_edge_distance = self.position[1]
        if nearest_shark_edge == shark_bottom:
            my_edge_distance = HEIGHT - self.position[1]

        for f in fishes:
            if f is self:
                continue
            if not f.alive:
                continue
            other_vector = np.array([
                f.position[0] - self.position[0],
                f.position[1] - self.position[1]
            ])
            other_distance = np.linalg.norm(other_vector)
            if other_distance < 1e-3:
                continue
            other_direction = (other_vector /other_distance)
            alignment = np.dot(other_direction,target_direction)
            other_edge_distance = 0
            if nearest_shark_edge == shark_left:
                other_edge_distance = f.position[0]
            if nearest_shark_edge == shark_right:
                other_edge_distance = (WIDTH - f.position[0])
            if nearest_shark_edge == shark_top:
                other_edge_distance =  f.position[1]
            if nearest_shark_edge == shark_bottom:
                other_edge_distance = (HEIGHT - f.position[1])

            if alignment > 0.5 and (other_edge_distance < my_edge_distance):
                teammates_between_me_and_edge += 1
        # Normalize
        teammates_between_normalized = teammates_between_me_and_edge / 44.0

        # Collect a Direction from Fish Brain
        outputs = brain.forward(left_edge_distance, right_edge_distance, top_edge_distance, bottom_edge_distance,
                                shark_direction_x, shark_direction_y,
                                nearest_edge_distance, shark_distance,
                                nearby_fish_density, fish_ahead_normalized,
                                closest_teammate_distance, teammates_between_normalized, relay_alignment,
                                relay_distance, shark_target_edge_distance)
        angle_output = outputs[0]
        angle_mean = angle_output * np.pi
        angle_distribution = torch.distributions.Normal(
            loc=angle_mean,
            scale=torch.tensor(0.3)
        )
        sample_angle = angle_distribution.sample()
        log_prob = angle_distribution.log_prob(sample_angle)
        direction_x = torch.cos(sample_angle)
        direction_y = torch.sin(sample_angle)

        self.log_probs.append(log_prob)
        self.took_action = True

        # Change Vector by Direction
        self.vector = [
            self.magnitude * direction_x.detach().numpy(),
            self.magnitude * direction_y.detach().numpy()
        ]
        self.position[0] += self.vector[0]
        self.position[1] += self.vector[1]

        # The Fish Die if They Get to the Edge
        if border.circle_collides(self.position, 7):
            self.alive = False
            self.edge_penalty = True

    def learn(self, hunted, won, shark_edge_change, shark_following, shark_position):
        self.reward = 0
        if self.alive:
            self.reward += 0.05
        # Benefit for Bait Fish
        ideal_distance = 120
        if shark_following:
            self.reward += 1.0 * shark_edge_change
            # Try to keep a reasonable distance
            bait_distance = np.hypot(
                self.position[0] - shark_position[0],
                self.position[1] - shark_position[1]
            )
            if 80 <= bait_distance <= 160:
                self.reward += 0.75
            # Too close is bad
            if bait_distance < 50:
                self.reward -= 0.75

        # Reward Relay Fish
        shark_to_fish_dx = self.position[0] - shark_position[0]
        shark_to_fish_dy = self.position[1] - shark_position[1]
        shark_to_fish_distance = np.hypot(shark_to_fish_dx, shark_to_fish_dy)
        alignment = np.dot(np.array([shark_to_fish_dx, shark_to_fish_dy]) / max(shark_to_fish_distance, 1e-3), self.target_direction)
        relay_distance = shark_to_fish_distance / np.hypot(WIDTH, HEIGHT)
        relay_score = 0.0
        if shark_to_fish_distance > 1e-3:
            relay_score = self.relay_alignment
        # Prefer being neither extremely close nor extremely far
        if 100 <= shark_to_fish_distance <= 300:
            relay_score *= 1.0
        else:
            relay_score *= 0.5

        if not shark_following and self.alive and self.relay_alignment > 0.5:
            self.reward += 0.5 * self.relay_alignment
            if 100 <= shark_to_fish_distance <= 300:
                self.reward += 0.75
            if 150 <= shark_to_fish_distance <= 250:
                self.reward += 0.5
            if shark_edge_change > 0:
                self.reward += 1.0 * shark_edge_change
        if not shark_following and self.alive and self.relay_alignment <= 0.5:
            self.reward += 0.05 * (np.hypot(shark_position[0] - self.position[0],
                                           shark_position[1] - self.position[1])  / np.hypot(WIDTH, HEIGHT))
        # Death Penalties
        if hunted:
            self.reward -= 30
        if self.edge_penalty:
            self.reward -= 30
            self.edge_penalty = False
        # Killing the shark is the objective
        if won:
            if self.alive:
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
        self.inputLayer = nn.Linear(15, 64)
        self.hiddenLayer1 = nn.Linear(64, 32)
        self.hiddenLayer2 = nn.Linear(32, 32)
        self.hiddenLayer3 = nn.Linear(32, 16)
        self.outputLayer = nn.Linear(16, 1) # Output an Angle
    # Inputs: Distance to Edge, Distance to Shark
        # Inputs: X Distance to Edge, Y Distance to Edge, X Distance to Shark, Y Distance to Shark
    def forward(self,left_edge_distance, right_edge_distance, top_edge_distance, bottom_edge_distance,
                                shark_direction_x, shark_direction_y,
                                nearest_edge_distance, shark_distance,
                                nearby_fish_density, fish_ahead_normalized,
                                closest_teammate_distance, teammates_between_normalized,
                                relay_alignment, relay_distance, shark_target_edge_distance):
        inputs = torch.tensor([left_edge_distance, right_edge_distance, top_edge_distance, bottom_edge_distance,
                                shark_direction_x, shark_direction_y,
                                nearest_edge_distance, shark_distance,
                                nearby_fish_density, fish_ahead_normalized,
                                closest_teammate_distance, teammates_between_normalized,
                               relay_alignment, relay_distance,shark_target_edge_distance],
                              dtype=torch.float32)
        hidden_layer_1 =  self.inputLayer(inputs)
        hidden_layer_1 = torch.tanh(hidden_layer_1)
        hidden_layer_2 = self.hiddenLayer1(hidden_layer_1)
        hidden_layer_2 = torch.tanh(hidden_layer_2)
        hidden_layer_3 = self.hiddenLayer2(hidden_layer_2)
        hidden_layer_3 = torch.tanh(hidden_layer_3)
        hidden_layer_4 = self.hiddenLayer3(hidden_layer_3)
        hidden_layer_4 = torch.tanh(hidden_layer_4)
        outputs = self.outputLayer(hidden_layer_4)
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
    target_edge = random.choice(['LEFT', 'RIGHT', 'TOP', 'BOTTOM'])
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
                fish.move(fish_brain, shark.position, collision, fishes)
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
    surviving_fish = sum(fish.alive for fish in fishes)
    survival_percentage = (surviving_fish / len(fishes)) * 100
    return fishes, quit_requested, frame_count, shark_died, survival_percentage

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
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (6, 7))
fig.subplots_adjust(hspace=1)

survival_line, = ax1.plot([], [], color = 'blue', marker = 'o', markersize = 2.3, linewidth = 1)
ax1.set_xlabel("Episode Number")
ax1.set_ylabel("Fish Survival (%)")
ax1.set_title("Fish Survival Rate")
ax1.set_ylim(0, 100)

death_rate_line, = ax2.plot([], [], color = 'red', linewidth = 0.7)
ax2.set_xlabel("Episode Number")
ax2.set_ylabel("Shark Death Rate (%)")
ax2.set_title("Cumulative Shark Death Rate")
ax2.set_ylim(0, 100)

reward_line, = ax3.plot([], [], color = 'green', marker='o', markersize = 2.3, linewidth = 1)
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
survival_rates = []
shark_died_flags = []
shark_death_rates = []
avg_rewards = []  # new

for i in range(5000):
    pygame.display.set_caption("Simulation " + str(i) + " of Mr. Fish Hunting")
    fishes, quit_requested, frame_count, shark_died, survival_percentage = run_instance(fish_brain)
    loss, avg_reward = train_instance(fishes, optimizer)  # now unpacks two values

    if quit_requested:
        break
    episode_numbers.append(i)
    shark_died_flags.append(shark_died)
    avg_rewards.append(avg_reward)
    survival_rates.append(survival_percentage)

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
    survival_line.set_data(episode_numbers, survival_rates)
    ax1.relim()
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, max(10, i + 1))

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

