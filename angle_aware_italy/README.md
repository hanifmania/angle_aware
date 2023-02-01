# angle_aware_italy
This is the code for https://www.sciencedirect.com/science/article/pii/S2405896322027847

## Demo


## Run Test
### Set config file
Without map
``` 
roscd angle_aware_italy/config
cp central_without_map.yaml central.yaml
```

from map
``` 
roscd angle_aware_italy/config
cp central_from_map.yaml central.yaml
```
### Simulation
```
roslaunch angle_aware_italy bringup_sim.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push X button -> main control start
1. Push START button -> take off and field reset

### Generate Psi (If you change phi, psi or z param, you should do this once)
Without map
```
roslaunch angle_aware_control generate_psi_without_map.launch
```
from map
```
roslaunch angle_aware_control generate_psi_from_map.launch
```

## Contents
- data
    - mat : Importance map from Italy CNR


## Author

[Takumi Shimizu](https://github.com/tashiwater)

