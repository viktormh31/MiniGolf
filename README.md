# Robotic RL Mini-golf
A robotic project utilizing an xArm 7 robotic arm to play a game of mini-golf. This project focuses on leveraging Reinforcement Learning and computer vision with a goal of bridging the gap between simulation and reality.

## Tech Stack
- Languages: Python
- Physics Engines: PyBullet, MuJoCo
- AI/ML: PyTorch, SAC + HER, YOLOv8
- Computer Vision: OpenCV
- OS/Infrastructure: Ubuntu 22.04, Git, ROCm (GPU Acceleration)

## Project Roadmap

### Simulation Phase - Complete :white_check_mark:
   This phase focused on developing a robust agent within simulated environment to prepare for hardware deployment:
- **Environment Modeling**: Developing simulated environment in **PyBullet**.
- **Policy Training**: Training an Reinforcement Learning (RL) agent using **SAC** and **HER** algorithms on ground-truth simulation data
- **Noise Injection**: Introduction of synthetic noise into the agent's observations to improve generalization and policy performance
- **Perception Integration**:
  - Integrating pretrained computer vision module - **YOLOv8**
  - Development of **custom OpenCV-based vision pipeline** to optimize performance.
- **Vision-Based Control**: Retraining RL policy using visual input from perception modules rather than absolute coordinates.
- **Optimization**: Extensive **hyperparameter fine-tuning** to reach a **90% success rate** on highest difficulty setting.

### Sim2Real Transfer - In Progress
   This project is currently transitioning to the physical xArm hardware. THe next steps involve:
- Architectural analysis:
  - Evaluation between the use of ROS2 framework and native xArm SDK
- Real-World Hardware Integration
  - Deployment of selected interface to the physical xArm
  - Testing trained policy on real hardware to identify differences between simulated and real-world performance
  - Adjusting the physics parameters like friction and latency to make simulation more accurate.



