"""
VYREX — RLBot v5 Inference Bot
================================
This file runs the trained VYREX neural network policy within RLBot v5.
It loads the trained POLICY.pt model and converts game state into actions.

Uses rlgym_compat to bridge RLBot v5 GamePacket → rlgym v2 GameState,
then builds observations with the exact same DefaultObs used in training.

Requirements:
    pip install rlbot torch numpy rlgym rlgym_compat
    Place POLICY.pt in this directory (exported from training).
"""

import os
import sys
import time
import traceback
import numpy as np

try:
    import torch
except ImportError:
    sys.path.insert(0, "../../torch-archive")
    import torch

import rlbot_flatbuffers as flat
from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot

from rlgym_compat import GameState
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league import common_values


# ============================================================================
# NEURAL NETWORK — Matches training architecture
# ============================================================================

class DiscreteFF(torch.nn.Module):
    """
    Feed-forward discrete policy network.
    Must match the architecture used during training (policy_layer_sizes).

    The LookupTableAction parser uses 90 discrete actions.
    """

    def __init__(self, input_size: int, output_size: int = 90,
                 layer_sizes=None):
        super().__init__()

        if layer_sizes is None:
            layer_sizes = [2048, 2048, 1024, 1024]

        layers = []
        prev_size = input_size
        for size in layer_sizes:
            layers.append(torch.nn.Linear(prev_size, size))
            layers.append(torch.nn.LeakyReLU())
            prev_size = size
        layers.append(torch.nn.Linear(prev_size, output_size))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================================
# VYREX BOT — Main RLBot v5 Bot Class
# ============================================================================

class VyrexBot(Bot):
    """
    VYREX — RLBot v5 bot that runs the trained PPO policy.

    Lifecycle:
      __init__()    — called immediately (no game data yet)
      initialize()  — called once field_info, match_config, index, team are ready
      get_output()  — called every tick with a GamePacket
    """

    POLICY_FILE = "POLICY.pt"
    DETERMINISTIC = True  # True = always pick best action, False = sample

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game_state = None
        self.obs_builder = None
        self.action_table = None
        self._obs_initialized = False
        self._tick_count = 0

        # Visualization state
        self._last_action_idx = 0
        self._last_confidence = 0.0
        self._last_entropy = 0.0
        self._last_top5_probs = []
        self._max_entropy = np.log(90)  # uniform over 90 actions
        self._render_enabled = True

    def initialize(self):
        """
        Called once when the bot is fully registered with RLBot and
        self.field_info, self.match_config, self.index, self.team,
        self.player_id are all available.
        """
        # --- Load the trained policy ---
        self._load_policy()

        # --- Create rlgym_compat GameState ---
        # This bridges RLBot v5 GamePacket → rlgym v2 compatible GameState
        self.game_state = GameState.create_compat_game_state(
            field_info=self.field_info,
            match_configuration=self.match_config,
        )

        # --- Create obs builder matching training EXACTLY ---
        # These coefficients MUST match build_vyrex_env() in train.py
        self.obs_builder = DefaultObs(
            zero_padding=None,  # Fixed 2v2, no padding
            pos_coef=np.asarray([
                1.0 / common_values.SIDE_WALL_X,   # 1/4096
                1.0 / common_values.BACK_NET_Y,     # 1/6000 (actually 5120 but matching common_values)
                1.0 / common_values.CEILING_Z,       # 1/2044
            ]),
            ang_coef=1.0 / np.pi,
            lin_vel_coef=1.0 / common_values.CAR_MAX_SPEED,   # 1/2300
            ang_vel_coef=1.0 / common_values.CAR_MAX_ANG_VEL, # 1/5.5
            boost_coef=1.0 / 100.0,
        )

        # --- Use the EXACT same action lookup table as training ---
        self.action_table = LookupTableAction.make_lookup_table()

        print(f"[VYREX] Initialized: index={self.index}, team={self.team}, "
              f"player_id={self.player_id}, actions={len(self.action_table)}")

    def _load_policy(self):
        """Load the trained policy network."""
        policy_path = os.path.join(os.path.dirname(__file__), self.POLICY_FILE)

        if not os.path.exists(policy_path):
            print(f"[VYREX] ERROR: Policy file not found: {policy_path}")
            print("[VYREX] Train the bot first, then run: python train.py --export")
            return

        try:
            state_dict = torch.load(policy_path, map_location=self.device, weights_only=True)

            # Infer input/output sizes from weights
            first_key = next(iter(state_dict))
            input_size = state_dict[first_key].shape[1]

            keys = list(state_dict.keys())
            last_weight_key = [k for k in keys if 'weight' in k][-1]
            output_size = state_dict[last_weight_key].shape[0]

            self.policy = DiscreteFF(
                input_size=input_size,
                output_size=output_size,
                layer_sizes=[2048, 2048, 1024, 1024],  # Must match training!
            )
            self.policy.load_state_dict(state_dict)
            self.policy.to(self.device)
            self.policy.eval()

            print(f"[VYREX] Policy loaded successfully ({input_size} → {output_size})")
            print(f"[VYREX] Device: {self.device}")

        except Exception as e:
            print(f"[VYREX] Failed to load policy: {e}")
            traceback.print_exc()
            self.policy = None

    def get_output(self, packet: GamePacket) -> ControllerState:
        """
        Called every tick by RLBot. Convert game state to actions.
        """
        controller = ControllerState()

        if self.policy is None or self.game_state is None:
            return controller

        try:
            # Update the rlgym_compat game state from the RLBot packet
            self.game_state.update(packet)

            # Our agent ID in the game_state.cars dict is player_id
            agent_id = self.player_id

            # Check our car exists in the state
            if agent_id not in self.game_state.cars:
                return controller

            # Reset obs builder on first tick (after cars are populated)
            if not self._obs_initialized:
                agents = list(self.game_state.cars.keys())
                self.obs_builder.reset(agents, self.game_state, {})
                self._obs_initialized = True

            # Build observation — returns {agent_id: np.ndarray}
            obs_dict = self.obs_builder.build_obs(
                [agent_id], self.game_state, {}
            )
            obs = obs_dict[agent_id]

            # Forward through the policy network
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.policy(obs_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze(0)

                if self.DETERMINISTIC:
                    action_idx = torch.argmax(probs).item()
                else:
                    action_idx = torch.multinomial(probs.unsqueeze(0), 1).item()

                # Capture visualization data
                self._last_action_idx = action_idx
                self._last_confidence = probs[action_idx].item()
                log_probs = torch.log(probs + 1e-10)
                self._last_entropy = -(probs * log_probs).sum().item()
                top5 = torch.topk(probs, min(5, len(probs)))
                self._last_top5_probs = list(zip(
                    top5.indices.cpu().tolist(),
                    top5.values.cpu().tolist(),
                ))

            # Convert discrete action index to game controls
            if action_idx < len(self.action_table):
                action = self.action_table[action_idx]
                controller.throttle = float(action[0])
                controller.steer = float(action[1])
                controller.pitch = float(action[2])
                controller.yaw = float(action[3])
                controller.roll = float(action[4])
                controller.jump = bool(action[5])
                controller.boost = bool(action[6])
                controller.handbrake = bool(action[7])

            # Draw debug visualization
            if self._render_enabled:
                self._draw_debug(packet, controller)

            # Periodic debug logging
            self._tick_count += 1
            if self._tick_count % 1000 == 1:
                print(f"[VYREX] tick={self._tick_count}, action={action_idx}, "
                      f"conf={self._last_confidence:.3f}, "
                      f"entropy={self._last_entropy:.2f}/{self._max_entropy:.2f}, "
                      f"obs_size={len(obs)}, cars={len(self.game_state.cars)}")

        except Exception as e:
            # Log errors instead of silently swallowing them
            print(f"[VYREX] ERROR in get_output: {e}")
            traceback.print_exc()

        return controller

    # ------------------------------------------------------------------
    # IN-GAME DEBUG VISUALIZATION
    # ------------------------------------------------------------------

    def _draw_debug(self, packet: GamePacket, ctrl: ControllerState):
        """Draw in-game debug overlays to verify the policy is intentional."""
        try:
            car = packet.players[self.index]
            car_pos = car.physics.location
            ball_pos = packet.balls[0].physics.location if packet.balls else None

            # Colors
            WHITE = flat.Color(r=255, g=255, b=255, a=255)
            CYAN = flat.Color(r=0, g=255, b=255, a=200)
            GREEN = flat.Color(r=0, g=255, b=0, a=200)
            YELLOW = flat.Color(r=255, g=255, b=0, a=200)
            RED = flat.Color(r=255, g=60, b=60, a=200)
            ORANGE = flat.Color(r=255, g=165, b=0, a=200)
            DIM = flat.Color(r=100, g=100, b=100, a=120)
            BG = flat.Color(r=0, g=0, b=0, a=160)

            # Entropy ratio: 0 = fully decisive, 1 = fully random
            entropy_ratio = self._last_entropy / self._max_entropy

            # Determine policy health color
            if entropy_ratio < 0.85:
                health_color = GREEN      # Decisive — the network has preferences
            elif entropy_ratio < 0.95:
                health_color = YELLOW     # Somewhat random
            else:
                health_color = RED        # Near-uniform = essentially random

            # --- Group for this bot instance ---
            group = f"vyrex_debug_{self.index}"
            self.renderer.begin_rendering(group)

            # HUD x offset: bot 0 on left, bot 1 on right
            hud_x = 0.01 if self.index == 0 else 0.75
            y = 0.03
            s = 1  # text scale

            # Header
            self.renderer.draw_string_2d(
                f"VYREX #{self.index}", hud_x, y, s + 0.5, CYAN, BG)
            y += 0.035

            # Confidence & Entropy
            conf_pct = self._last_confidence * 100
            rand_pct = (1.0 / 90) * 100  # 1.11% = random baseline
            is_intentional = conf_pct > rand_pct * 2  # >2x random

            self.renderer.draw_string_2d(
                f"Confidence: {conf_pct:5.1f}%  (random={rand_pct:.1f}%)",
                hud_x, y, s, health_color, BG)
            y += 0.025

            self.renderer.draw_string_2d(
                f"Entropy: {self._last_entropy:.3f} / {self._max_entropy:.3f} "
                f"({entropy_ratio*100:.1f}%)",
                hud_x, y, s, health_color, BG)
            y += 0.025

            # Verdict
            if entropy_ratio < 0.85:
                verdict = "POLICY IS INTENTIONAL"
                verdict_color = GREEN
            elif entropy_ratio < 0.95:
                verdict = "POLICY SEMI-RANDOM"
                verdict_color = YELLOW
            else:
                verdict = "POLICY IS RANDOM!"
                verdict_color = RED
            self.renderer.draw_string_2d(
                verdict, hud_x, y, s + 0.3, verdict_color, BG)
            y += 0.035

            # Top 5 actions
            self.renderer.draw_string_2d(
                "Top Actions:", hud_x, y, s, WHITE, BG)
            y += 0.022
            for i, (aidx, prob) in enumerate(self._last_top5_probs):
                bar_len = int(prob * 200)  # scale
                bar = "|" * min(bar_len, 30)
                mark = " <<" if aidx == self._last_action_idx else ""
                self.renderer.draw_string_2d(
                    f"  [{aidx:2d}] {prob*100:5.1f}% {bar}{mark}",
                    hud_x, y, s * 0.85, CYAN if i == 0 else DIM, BG)
                y += 0.018

            y += 0.01

            # Current controls
            self.renderer.draw_string_2d(
                f"Throttle:{ctrl.throttle:+.0f}  Steer:{ctrl.steer:+.0f}  "
                f"Boost:{'ON' if ctrl.boost else '--'}  "
                f"Jump:{'ON' if ctrl.jump else '--'}  "
                f"Slide:{'ON' if ctrl.handbrake else '--'}",
                hud_x, y, s * 0.9, ORANGE, BG)
            y += 0.025

            self.renderer.draw_string_2d(
                f"Pitch:{ctrl.pitch:+.0f}  Yaw:{ctrl.yaw:+.0f}  "
                f"Roll:{ctrl.roll:+.0f}  Action:{self._last_action_idx}",
                hud_x, y, s * 0.9, ORANGE, BG)

            # --- 3D: Line from car to ball ---
            if ball_pos:
                car_anchor = flat.CarAnchor(
                    index=self.index,
                    local=flat.Vector3(x=0, y=0, z=50)
                )
                ball_anchor = flat.BallAnchor(
                    local=flat.Vector3(x=0, y=0, z=0)
                )
                line_color = GREEN if is_intentional else RED
                self.renderer.draw_line_3d(car_anchor, ball_anchor, line_color)

            # --- 3D: Label above car ---
            car_label_anchor = flat.CarAnchor(
                index=self.index,
                local=flat.Vector3(x=0, y=0, z=120)
            )
            self.renderer.draw_string_3d(
                f"VYREX  {conf_pct:.0f}%",
                car_label_anchor, 1.0, CYAN, BG)

            self.renderer.end_rendering()

        except Exception as e:
            # Never let rendering crash the bot
            if self._tick_count % 500 == 0:
                print(f"[VYREX] Render error: {e}")


# ============================================================================
# ENTRY POINT — Required by RLBot v5
# ============================================================================

if __name__ == "__main__":
    bot = VyrexBot("vyrex/vyrex-2v2")
    bot.run()
