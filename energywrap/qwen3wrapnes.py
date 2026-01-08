"""
Curiosity-driven RL system using Qwen3-VL with Energy Dynamics to play Super Mario Bros NES.

Uses gym_super_mario_bros with the working API from the reference implementation.

The energy dynamics act as an intrinsic curiosity mechanism, where:
- High energy neurons fire when encountering novel states
- Low energy neurons shut off for repetitive patterns
- Auxiliary loss drives exploration through sparsity
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import logging
from collections import deque
import time

# Import displayarray for visualization
from displayarray import display

from qwen3vlwrap import load_qwen3vl_with_energy_dynamics
from manual_energy_wrapper import EnergyWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarioVisionWrapper:
    """Wrapper to process Mario frames for Qwen3-VL."""

    def __init__(self, env):
        self.env = env
        self.frame_buffer = deque(maxlen=4)

        # Detect if env returns 4 or 5 values
        self.env.reset()
        test_result = self.env.step(0)
        self.returns_five = len(test_result) == 5
        self.env.reset()

    def reset(self):
        result = self.env.reset()
        # Handle both old gym (returns obs) and new gymnasium (returns obs, info)
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result

        self.frame_buffer.clear()
        for _ in range(4):
            self.frame_buffer.append(obs)
        return obs

    def step(self, action):
        result = self.env.step(action)

        if self.returns_five:
            # New gymnasium API: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            # Old gym API: obs, reward, done, info
            obs, reward, done, info = result

        self.frame_buffer.append(obs)
        return obs, reward, done, info

    def get_stacked_frame(self):
        """Get current frame for vision model."""
        return np.array(self.frame_buffer[-1])

    def close(self):
        self.env.close()

    def seed(self, seed):
        try:
            self.env.seed(seed)
            self.env.action_space.seed(seed)
        except:
            pass


class CuriosityRLAgent:
    """RL agent using Qwen3-VL with energy dynamics as curiosity mechanism."""

    def __init__(
        self,
        model: EnergyWrapper,
        processor,
        action_space_size: int,
        #device: str = "cuda",
        device: str = "cpu",
        curiosity_weight: float = 0.5,
        learning_rate: float = 1e-5,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.action_space_size = action_space_size
        self.curiosity_weight = curiosity_weight

        # Action head - maps model output to actions
        # Access the underlying config via the wrapped model
        hidden_size = model.model.config.text_config.hidden_size
        self.action_head = torch.nn.Linear(hidden_size, action_space_size).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.action_head.parameters()),
            lr=learning_rate
        )

        # Experience buffer
        self.experience_buffer = []
        self.max_buffer_size = 1000

        # Statistics
        self.episode_rewards = []
        self.episode_curiosity = []
        self.episode_x_pos = []
        self.action_names = [
            "NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A", "LEFT"
        ]

    def frame_to_pil(self, frame):
        """Convert game frame to PIL Image."""
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if len(frame.shape) == 2:  # Grayscale
            frame = np.stack([frame] * 3, axis=-1)
        return Image.fromarray(frame)

    def get_action_logits(self, frame):
        """Get action logits from vision input."""
        # Convert frame to PIL
        pil_frame = self.frame_to_pil(frame)

        # Prepare prompt - minimal context as requested
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_frame},
                    {"type": "text", "text": "You are playing super mario for the NES"},
                ],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[pil_frame],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # EnergyWrapper forwards the call to the underlying model
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get last hidden state and map to actions
        hidden_state = outputs.hidden_states[-1][:, -1, :]  # Last token
        action_logits = self.action_head(hidden_state)

        # Get curiosity signal from energy dynamics using the wrapper
        # The wrapper aggregates aux_loss from all hooks
        curiosity_reward = self.model.get_total_aux_loss()

        # Ensure it's a float for the RL reward signal
        if isinstance(curiosity_reward, torch.Tensor):
            curiosity_reward = curiosity_reward.item()

        return action_logits, curiosity_reward

    def select_action(self, frame, epsilon=0.1):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            # Random exploration
            action = np.random.randint(0, self.action_space_size)
            curiosity = 0.0
        else:
            # Policy action
            action_logits, curiosity = self.get_action_logits(frame)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs, dim=-1).item()

        return action, curiosity

    def store_experience(self, state, action, reward, next_state, done, curiosity):
        """Store experience in replay buffer."""
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'curiosity': curiosity
        })

        # Keep buffer size manageable
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def update_policy(self, batch_size=4):
        """Update policy using stored experiences."""
        if len(self.experience_buffer) < batch_size:
            return None

        # Sample batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        total_loss = 0
        for exp in batch:
            # Get action logits
            action_logits, curiosity = self.get_action_logits(exp['state'])

            # Compute policy loss (REINFORCE-style)
            action_probs = F.log_softmax(action_logits, dim=-1)
            action_log_prob = action_probs[0, exp['action']]

            # Total reward = external + curiosity
            total_reward = exp['reward'] + self.curiosity_weight * exp['curiosity']

            # Policy gradient loss
            policy_loss = -action_log_prob * total_reward

            # Energy dynamics loss (from wrapper)
            energy_loss = self.model.get_total_aux_loss()

            # Combined loss
            loss = policy_loss + 0.1 * energy_loss
            total_loss += loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item()

    def play_episode(self, env, max_steps=500, epsilon=0.1, train=True, displayer=None):
        """
        Play one episode of Mario.
        If displayer is provided, visualization is updated in real-time.
        """
        state = env.reset()
        episode_reward = 0
        episode_curiosity = 0
        steps = 0
        max_x_pos = 0

        logger.info(f"Starting episode (epsilon={epsilon:.3f}, train={train})")

        # Initial step to get info
        state, _, _, info = env.step(0)

        while steps < max_steps:
            # Select action
            action, curiosity = self.select_action(state, epsilon)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Update Visualization
            if displayer:
                # 1. Prepare Game Frame
                vis_frame = state.copy()

                # 2. Prepare Energy Heatmap (Visualizing the "Curiosity")
                vis_energy = None
                try:
                    # Get the energy tensor from the last hooked layer (closest to decision)
                    if hasattr(self.model, 'hooked_layers') and self.model.hooked_layers:
                        # Grab last layer's module
                        _, last_layer_mod = self.model.hooked_layers[-1]
                        if last_layer_mod.energy is not None:
                            # Detach and move to cpu
                            e = last_layer_mod.energy.detach().cpu().float().numpy()

                            # Normalize Energy: Typically ranges [-2, 2]
                            # Map to [0, 1] for visualization
                            e_norm = (e + 2.0) / 4.0
                            e_norm = np.clip(e_norm, 0.0, 1.0)

                            # Reshape 1D energy to 2D square for heatmap
                            # Calculate side length
                            side = int(np.ceil(np.sqrt(e_norm.size)))
                            # Pad to fit square
                            pad_size = side * side - e_norm.size
                            if pad_size > 0:
                                e_norm = np.pad(e_norm, (0, pad_size), mode='constant')

                            vis_energy = e_norm.reshape(side, side)
                            # Stack to RGB for displayarray
                            vis_energy = np.stack([vis_energy] * 3, axis=-1)
                except Exception as e:
                    # Fail silently on vis errors to keep training running
                    pass

                # Update the display
                if vis_energy is not None:
                    # Show Mario and the Neural Energy state side-by-side
                    displayer.update([vis_frame, vis_energy], ["Mario Gameplay", "Neural Energy (Curiosity)"])
                else:
                    displayer.update(vis_frame, ["Mario Gameplay"])

            # Track max x position
            x_pos = info.get('x_pos', 0)
            max_x_pos = max(max_x_pos, x_pos)

            # Store experience
            if train:
                self.store_experience(state, action, reward, next_state, done, curiosity)

            # Update stats
            episode_reward += reward
            episode_curiosity += curiosity
            steps += 1

            # Log progress
            if steps % 50 == 0:
                logger.info(
                    f"Step {steps}: Action={self.action_names[action]}, "
                    f"Reward={reward:.2f}, Curiosity={curiosity:.4f}, "
                    f"X-pos={x_pos}"
                )

            state = next_state

            if done:
                break

            # Check if user closed the display window
            if displayer and not displayer:
                logger.info("Visualization window closed by user.")
                return episode_reward, episode_curiosity, max_x_pos

        # Update policy
        if train and len(self.experience_buffer) >= 4:
            loss = self.update_policy(batch_size=4)
            if loss:
                logger.info(f"Policy updated: loss={loss:.4f}")

        self.episode_rewards.append(episode_reward)
        self.episode_curiosity.append(episode_curiosity)
        self.episode_x_pos.append(max_x_pos)

        logger.info(
            f"Episode finished: {steps} steps, "
            f"reward={episode_reward:.2f}, curiosity={episode_curiosity:.4f}, "
            f"max x-pos={max_x_pos}"
        )

        return episode_reward, episode_curiosity, max_x_pos


def create_mario_env():
    """Create Super Mario Bros environment."""
    # Create base environment - gym_super_mario_bros uses old gym API
    env = gym_super_mario_bros.make('SuperMarioBros-v3')

    # Wrap with JoypadSpace for simplified controls
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Add vision wrapper
    env = MarioVisionWrapper(env)

    return env


def train_curiosity_agent(
    num_episodes=100,
    max_steps_per_episode=500,
    epsilon_start=0.3,
    epsilon_end=0.05,
    epsilon_decay=0.995,
):
    """Train the curiosity-driven RL agent."""
    logger.info("Loading Qwen3-VL with energy dynamics...")

    # Load base model with energy hooks
    raw_model, processor = load_qwen3vl_with_energy_dynamics(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        #device="cuda" if torch.cuda.is_available() else "cpu",
        device = "cpu",
        torch_dtype=torch.float32,
        gamma=0.05,  # Energy drain - encourages forgetting repetitive patterns
        lambda_kl=0.01,  # Sparsity - drives selective attention
        lambda_l1=0.005,
        beta=0.05,  # Target 5% active neurons - encourages specialization
    )

    # WRAP THE MODEL with EnergyWrapper to orchestrate hooks and losses
    logger.info("Wrapping model with EnergyWrapper...")
    model = EnergyWrapper(raw_model)

    logger.info("Creating Mario environment...")
    env = create_mario_env()

    # Set seed for determinism
    env.seed(42)

    logger.info("Initializing curiosity RL agent...")
    agent = CuriosityRLAgent(
        model=model,
        processor=processor,
        action_space_size=len(SIMPLE_MOVEMENT),
        #device="cuda" if torch.cuda.is_available() else "cpu",
        device = "cpu",
        curiosity_weight=0.5,  # Balance external and intrinsic rewards
        learning_rate=1e-5,
    )

    logger.info(f"Starting training for {num_episodes} episodes...")
    epsilon = epsilon_start

    # Initialize visualization context
    # We create a dummy frame to initialize the window
    dummy_frame = np.zeros((240, 256, 3), dtype=np.uint8)

    logger.info("Launching Visualization Window...")
    with display(dummy_frame) as displayer:

        for episode in range(num_episodes):
            if not displayer:
                logger.info("Display window closed. Stopping training loop.")
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"EPISODE {episode + 1}/{num_episodes}")
            logger.info(f"{'='*60}")

            # Play episode with displayer passed down
            episode_reward, episode_curiosity, max_x_pos = agent.play_episode(
                env,
                max_steps=max_steps_per_episode,
                epsilon=epsilon,
                train=True,
                displayer=displayer
            )

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Print statistics
            if len(agent.episode_rewards) >= 10:
                avg_reward = np.mean(agent.episode_rewards[-10:])
                avg_curiosity = np.mean(agent.episode_curiosity[-10:])
                avg_x_pos = np.mean(agent.episode_x_pos[-10:])
                logger.info(f"\nRecent performance (last 10 episodes):")
                logger.info(f"  Average reward: {avg_reward:.2f}")
                logger.info(f"  Average curiosity: {avg_curiosity:.4f}")
                logger.info(f"  Average max X-pos: {avg_x_pos:.1f}")

            # Save checkpoint periodically
            if (episode + 1) % 10 == 0:
                checkpoint_path = f"mario_agent_ep{episode+1}.pt"
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'action_head_state_dict': agent.action_head.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'episode_rewards': agent.episode_rewards,
                    'episode_curiosity': agent.episode_curiosity,
                    'episode_x_pos': agent.episode_x_pos,
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

    env.close()
    logger.info("\nTraining complete!")

    return agent, model


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("CURIOSITY-DRIVEN RL: Qwen3-VL plays Super Mario Bros")
    logger.info("Energy Dynamics = Intrinsic Curiosity Mechanism")
    logger.info("="*60)

    # Train agent
    agent, model = train_curiosity_agent(
        num_episodes=50,
        max_steps_per_episode=500,
        epsilon_start=0.3,
        epsilon_end=0.05,
    )

    logger.info("\nFinal Statistics:")
    logger.info(f"Total episodes: {len(agent.episode_rewards)}")
    logger.info(f"Average reward: {np.mean(agent.episode_rewards):.2f}")
    logger.info(f"Average curiosity: {np.mean(agent.episode_curiosity):.4f}")
    logger.info(f"Average max X-pos: {np.mean(agent.episode_x_pos):.1f}")
    logger.info(f"Best episode reward: {max(agent.episode_rewards):.2f}")
    logger.info(f"Best X-pos reached: {max(agent.episode_x_pos):.1f}")