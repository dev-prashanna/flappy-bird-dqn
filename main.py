"""Play manually or watch a trained PPO policy in the identical simulator."""
import argparse
from pathlib import Path

import numpy as np
import torch

from flappyenv import EnvConfig, FlappyBirdEnv
from ppo import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai", action="store_true", help="Watch a saved PPO agent")
    parser.add_argument("--model", default="models/ppo.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions instead of choosing the largest logit")
    parser.add_argument("--fps", type=int, default=60, help="Playback speed in simulation frames per second")
    parser.add_argument("--frames", type=int, default=None, help="Exit after N display frames (useful for a smoke test)")
    args = parser.parse_args()
    if args.fps <= 0 or (args.frames is not None and args.frames <= 0):
        parser.error("fps and frames must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    network, metadata = None, {}
    if args.ai:
        if not Path(args.model).is_file():
            parser.error(f"No PPO checkpoint at {args.model}. Run: python train_net.py")
        network, metadata = load_policy(args.model)
    from renderer import Renderer, pygame
    config = EnvConfig(**metadata.get("env_config", {}))
    env = FlappyBirdEnv(config)
    observations, _ = env.reset(seed=args.seed)
    renderer = Renderer(config)
    running, started, paused, done = True, args.ai, False, False
    displayed = 0
    try:
        while running:
            flap = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        observations, _ = env.reset()
                        done, started, paused = False, args.ai, False
                    elif event.key == pygame.K_p and started and not done:
                        paused = not paused
                    elif event.key in (pygame.K_SPACE, pygame.K_UP):
                        if done:
                            observations, _ = env.reset()
                            done = False
                        started, flap = True, True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if done:
                        observations, _ = env.reset()
                        done = False
                    started, flap = True, True
            if not running:
                break
            if started and not paused and not done:
                action = int(flap)
                if network is not None:
                    with torch.no_grad():
                        action = int(network.action_value(torch.as_tensor(np.asarray(observations))[None],
                                                          deterministic=not args.stochastic)[0].item())
                observations, _, term, trunc, _ = env.step(action)
                done = term or trunc
            overlay = None
            if done:
                overlay = [f"Flight complete: {env.score} pipes", "SPACE / click to retry", "R  restart    ESC  exit"]
            elif paused:
                overlay = ["Paused", "P  resume"]
            elif not started:
                overlay = ["Ready for takeoff", "SPACE / UP / click to flap", "P  pause    R  restart    ESC  exit"]
            mode = "PPO / sampled actions" if args.stochastic else "PPO / greedy actions"
            subtitle = (mode if args.ai else "Manual flight") + "     P: pause   R: restart   ESC: exit"
            renderer.draw(env, subtitle, overlay)
            renderer.clock.tick(args.fps)
            displayed += 1
            if args.frames and displayed >= args.frames:
                running = False
    finally:
        env.close()
        renderer.close()


if __name__ == "__main__":
    main()
