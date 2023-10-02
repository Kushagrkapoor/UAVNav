# UAVNav
This is a sample project which contains some general elements of a graduate research project I undertook at Purdue ECE. In the original project, DDPG was used to select optimal actions. Rigorous validation and simulations were performed using the CARLA simulator. 

Due to the project not being open source yet, I use GymAI here to generate environments and use similar (but not the exact) algorithmic techniques that were used during the project. 

To be precise, here is what I've done:
  1. Generated environments from GymAI.
  2. Created an actor-critic network loop to execute and evaluate Q Values of actions.
  3. To approximate Q Values, DDPG was used since Action space was continuous. 

![alt text](https://github.com/Kushagrkapoor/UAVNav/assets/48654665/deb44a9a-462d-4637-9000-1011617ab308)

Image courtesy: https://carla.readthedocs.io/en/0.9.11/core_map/
