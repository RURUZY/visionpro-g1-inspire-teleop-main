# Proposal: Language-Conditioned Tool Use with Tactile Slip Monitoring on a G1 Humanoid

**Platform:** Unitree G1 + Inspire five-finger hands (model to be confirmed), Apple Vision Pro teleoperation, MuJoCo simulation
**Status date:** 17 October 2026. All results below are from simulation; nothing has been validated on hardware yet.

## 1. Idea

A humanoid uses a hand-held tool (a sponge) to wipe a table region named in a language instruction ("wipe the left area" / "wipe the right area"). Two parts:

1. **Policy.** A language-conditioned imitation-learning policy (ACT baseline → language-conditioned ACT; optionally fine-tuning a pretrained VLA) maps head-camera images, joint states and the instruction to arm and finger commands.
2. **Execution monitoring.** A tactile slip detector watches the grasp and triggers a corrective action (e.g. increase grip force) when the tool starts to slip in the hand.

The project combines the two posted topics *Tool-Use VLA for Humanoid Hands* and *Tactile Sensing for Dexterous Robotic Hands*: the tactile module is the "execution monitoring / tactile feedback" component of the tool-use pipeline.

## 2. Proposed scope

- **Core (Bachelor scope):**
  - One tool (sponge), one task family.
  - A dataset and closed-loop benchmark.
  - Evaluation of an existing policy (ACT / language-conditioned ACT).
  - A tactile slip detector with a with-vs-without feedback comparison.
- **Extension (Master scope, if applicable):**
  - Contact-aware policy: tactile signals as an input, or a hierarchical controller with slip monitoring.
  - Transfer across tool variants (handled sponge vs. palm-held sponge block, mass and friction changes).

Evaluation metrics:

- task success with 95% confidence intervals;
- target vs. non-target coverage;
- tool drops and slip events;
- fingertip force;
- detection latency;
- generalization to rephrased or swapped instructions and changed tool parameters.

## 3. What already works (simulation)

- **Teleoperation:** a sponge-wiping scene running on the same Vision Pro → SONIC whole-body-control pipeline as our existing cube-grasping setup.
- **Scripted demonstration generator** that writes LeRobot v3 datasets:
  - head-camera RGB, 13-D arm+hand state and action, language instruction, 20 Hz;
  - per-episode success labels.
- **Tactile recording:**
  - 17 contact pads on the simulated hand, matching the 17 tactile groups of the Inspire FTP hand;
  - finger actuator forces;
  - randomized grip, mass, friction and disturbances, so that about 40% of episodes contain real in-hand slip.
- **Slip detector** using only tactile and finger-force signals, tested on held-out episodes:
  - 40% event recall at 75 ms median latency and 0.6 false alarms per minute;
  - a simple force-change threshold gets 20%.
- **First feedback experiment:** no improvement yet. Success was 82% with feedback vs. 85% without and 93% with constant maximum grip (27 matched episodes each; differences are within the confidence intervals). The detector fires too rarely and too late. This is the baseline the thesis would improve on.
- **Closed-loop evaluator** for trained policies. The pipeline data → training → evaluation has been verified end-to-end with a short ACT training run on CPU.

## 4. Feasibility and risks

| Risk | Mitigation |
|---|---|
| Real hand model and tactile availability unknown | Tactile code is isolated in one module; confirm the hand in week 1. Without tactile sensing, the project falls back to finger force/current signals. |
| Contact-only grasping is fragile in simulation | Use the handled tool (holds by contact); keep the palm-held variant as an explicitly labeled assisted-grip baseline. |
| Sim-to-real gap for tactile signals | Treat simulation as method development; re-fit detector thresholds on real tactile data; report real-robot results separately. |
| Training compute | See section 5. |
| Limited real-robot time | Simulation work proceeds in parallel; real-robot sessions focus on control checks, sensor calibration and teleoperated demonstrations. |

## 5. Compute request

The Jetson Orin on the robot can run data generation and evaluation, but not training: ACT (52 M parameters) takes about 4.5 s per training step on its CPU, so a 20k-step run would take more than a day.

Estimated needs (to be measured on the first runs):

| Use | Suggested GPU | Approx. time |
|---|---|---|
| ACT / language-conditioned ACT, single camera, 200–400 episodes | 1 GPU with ≥ 24 GB (e.g. RTX 4090 / A5000 / A6000) | a few hours per run, 10–20 runs |
| Optional: fine-tuning a pretrained VLA (~3 B parameters, e.g. π0 or GR00T N1.x) | 1× A100/H100 80 GB, or ≥ 48 GB with parameter-efficient fine-tuning | ~1–2 days per run |

Access to a lab workstation or the university cluster for the ACT stage would be sufficient to start. The VLA stage is optional and can be dropped if large-GPU access is not available.

## 6. Plan (8–12 weeks)

| Weeks | Work |
|---|---|
| 1–2 | Confirm the hand and its tactile interface; real-robot control and teleoperation checks; ACT baseline in simulation |
| 3–4 | Language-conditioned ACT in simulation; improved slip detector (more data, temporal model) and feedback actions |
| 5–6 | Real teleoperated demonstrations; tactile calibration on hardware; fine-tune on real data |
| 7–8 | Real-robot evaluation: success, slips, forces, with vs. without tactile feedback; write-up (core scope) |
| 9–12 | Extension: contact-aware or hierarchical policy, transfer across tool variants (Master scope) |

## 7. Questions for the supervisor

1. Is the combined topic (tool-use policy + tactile slip monitoring) acceptable, and at which level (Bachelor or Master)?
2. Which dexterous hand will be available on the real robot, and does it provide tactile data?
3. Can we get GPU access (lab workstation or cluster) for the ACT stage, and possibly a large GPU for a VLA fine-tuning stage?
4. How much real-robot time per week can be planned?
