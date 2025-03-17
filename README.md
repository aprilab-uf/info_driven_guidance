# Code for the "Uncertainty-Aware Guidance for Target Tracking subject to Intermittent Measurements using Motion Model Learning" paper

![Motion Model Learning](media/mml_acc.gif)

## Requirements
- Run `pip3 install -r requirements.txt` to install all the required packages. You also need the following repositories.
- [Reef estimator simulation](https://github.com/uf-reef-avl/reef_estimator_sim_bundle).
	- You need to make sure all the submodules are up to date and tracking master branch. Do `git submodule update --init --recursive` to update the submodules.
	- To track master for all submodules, type `git submodule foreach --recursive git checkout master`
	- Note: to automatically change the number of turtlebots, inside the `sim_helper` repo, type `git remote set-url origin https://github.com/andrespulido8/sim_helper.git. Then `git checkout mml`
	- If you want to run only hardware then you only need reef_estimator
- [Turtlebot packages](https://automaticaddison.com/how-to-launch-the-turtlebot3-simulation-with-ros/#gazebo) for the turtlebot simulation.
- [Andres turtlebot PID controller](http://10.251.72.180/andres/andres_turtlebot_pid) for turtlebot controller.
- [RosFlight](https://github.com/uf-reef-avl/torque_flight) for quadcopter autopilot.

## Sim Usage

1. To adjust the number of simulated vehicles modify the `./launch/launch_sim.launch` file inside the sim_helper repository.
2. Change the `spawn_turtles` argument inside the previously mentioned launch file to `robot0`.
3. In the `sim_helper` repository run `python scripts/Master.py` to start the simulation.
4. Wait a few seconds until __Autopilot ARMED__ and __RC override active__ are printed and then in another terminal
run `roslaunch mml_guidance mml_sim_estimator.launch` from the launch directory.
NOTE: To visualize the particle filter and the motion model, run `roslaunch mml_guidance visualization.launch` instead of `mml_sim_estimator.launch`.

## Hardware Usage
To run only the turtlebot, do `roslaunch mml_guidance turtlebot_hardware.launch`.
To run only the quadcopter, do `roslaunch mml_guidance track_hardware.launch`.
To bag data during hardware experiments run `roslaunch mml_guidance bag_hardware.launch prefix_name:="<insert prefix>"`

## Motion Model Learning (Neural Network)
The files needed to run the NN are located in [this DropBox](https://www.dropbox.com/sh/dmmskhd9mjbo9ws/AAD5oRf90joVTDinnghFxzG7a?dl=0).
You should move the csv to `mml_guidance/scripts/mml_network/` and the .pth weights to `mml_guidance/scripts/mml_network/models/`.
The DropBox also has the data used in the results of the letter.

## Train
To turn off the Gazebo GUI to make the sim faster, change the argument `gui` to `false` in `camera_multirotor.launch`
inside the **sim_helper** package from REEF github

## Contributing Guide
To make changes to this repo, it is recommended to use the tool [pre-commit](https://pre-commit.com/).
To install it, run `pip3 install -r requirements.txt` inside this repo, and then install the hooks
specified in the config file by doing `pre-commit install`. Now to run it against all the files to check
if it worked, run `pre-commit run --all-files`.

## Profiling
Run `roslaunch mml_guidance mml_sim_estimator.launch` and then
`pprofile --format callgrind --out guidance.pprofile /home/andrespulido/catkin_ws/src/mml_guidance/scripts/guidance.py __name:=drone_guidance`.
This will run the profiler and save the results in the directory where the command was called.

### Extra
Install [**tmuxinator**](https://github.com/tmuxinator/tmuxinator) to easily run the sim. Use `sudo apt-get install -y tmuxinator`. Then to run the `tmuxinator` command in the /mml_guidance directory. You can change the layout of the sessions with `ctrl-b space`
