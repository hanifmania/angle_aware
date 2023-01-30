# angle_aware_italy
This is the code for https://www.sciencedirect.com/science/article/pii/S2405896322027847

## Demo


## Run Test
### Simulation
```
roslaunch angle_aware_italy bringup_sim.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push X button -> main control start
1. Push START button -> take off and field reset

### Generate Psi
If you can use JAX, then launch the following file. This is twice faster than generate_psi_no_jax.launch.
```
roslaunch angle_aware_control generate_psi.launch
```


## Contents
- angle_aware_control :
    - agent.py :  圧縮した重要度psiを受け取ってangle aware CBFに従いドローンを操作する
    - angle_aware_cbf.py : Angle Aware CBFのアルゴリズム
    - myqp.py : collision avoidance 等も踏まえた最終的なQP
    - central.py : 圧縮した重要度psiを更新し、publish
    - psi_generator.py : phi -> psiの圧縮. JAXを利用



## Author

[Takumi Shimizu](https://github.com/tashiwater)

