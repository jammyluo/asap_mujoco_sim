## B站：飞的岛
# 微信：  feidedaoRobot
## refer
##https://github.com/LeCAR-Lab/ASAP
##https://github.com/engineai-robotics/engineai_legged_gym
##https://github.com/unitreerobotics/unitree_rl_gym
 

import mujoco, mujoco_viewer  # pip install mujoco-python-viewer
import numpy as np
import onnxruntime
import yaml
import os
import time
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace

 
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def read_conf(config_file):
    cfg = SimpleNamespace()
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

     # get_obs:
    cfg.num_single_obs = config["num_single_obs"]
    cfg.simulation_dt = config["simulation_dt"]
    cfg.cycle_time = config["cycle_time"]
    cfg.frame_stack = config["frame_stack"]


    cfg.default_dof_pos = np.array(config["default_dof_pos"], dtype=np.float32)

    cfg.obs_scale_base_ang_vel = config["obs_scale_base_ang_vel"]
    cfg.obs_scale_dof_pos = config["obs_scale_dof_pos"]
    cfg.obs_scale_dof_vel = config["obs_scale_dof_vel"]
    cfg.obs_scale_gvec = config["obs_scale_gvec"]
    cfg.obs_scale_refmotion = config["obs_scale_refmotion"]
    cfg.obs_scale_hist = config["obs_scale_hist"]
 
    cfg.clip_observations = config["clip_observations"]
 
    
    
    #pd_control:
    cfg.kps = np.array(config["kps"], dtype=np.float32)
    cfg.kds = np.array(config["kds"], dtype=np.float32)

    #run_mujoco:
    cfg.xml_path = config["xml_path"]
    cfg.num_actions = config["num_actions"]
    cfg.policy_path = config["policy_path"]
    cfg.simulation_duration = config["simulation_duration"]
    cfg.control_decimation = config["control_decimation"]
    cfg.clip_actions = config["clip_actions"]
    cfg.action_scale = config["action_scale"]
    cfg.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
 
    return cfg
     
def get_mujoco_data(data):
    mujoco_data={}
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = np.array([q[4], q[5], q[6], q[3]])
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    base_angvel = dq[3:6]
    # line_acc = data.sensor('imu-linear-acceleration').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    
    # import math
    # root_euler = quaternion_to_euler_array(quat)
    # root_euler[root_euler > math.pi] -= 2 * math.pi

    mujoco_data['mujoco_dof_pos'] = q[7:]
    mujoco_data['mujoco_dof_vel'] = dq[6:]
    mujoco_data['mujoco_base_angvel'] = base_angvel
    mujoco_data['mujoco_gvec'] = gvec
    return mujoco_data
 
def update_hist_obs(hist_dict, obs_sigle):
    '''
        history_keys = ['actions', 'base_ang_vel', 'dof_pos',
                    dof_vel', 'projected_gravity', 'ref_motion_phase']
    '''
    slices = {
        'actions': slice(0, 23),
        'base_ang_vel': slice(23, 26),
        'dof_pos': slice(26, 49),
        'dof_vel': slice(49, 72),
        'projected_gravity': slice(72, 75),
        'ref_motion_phase': slice(75, 76)
    }
    
    for key, slc in slices.items():
        # Remove oldest entry and add new observation
        arr = np.delete(hist_dict[key], -1, axis=0)
        arr = np.vstack((obs_sigle[0, slc], arr))
        hist_dict[key] = arr
    
    hist_obs = np.concatenate([
        hist_dict[key].reshape(1, -1) 
        for key in hist_dict.keys()
    ], axis=1).astype(np.float32)
    return hist_obs

def get_obs(hist_obs_c,hist_dict,mujoco_data,action,counter,cfg):
    ''' obs:
    action #  23
    base_ang_vel # 3
    dof_pos # 23
    dof_vel # 23
    history_actor # 4 * (23+3+23+23+3+1)=4*76=304
    projected_gravity # 3
    ref_motion_phase # 1 
    '''
    mujoco_base_angvel = mujoco_data["mujoco_base_angvel"]
    mujoco_dof_pos = mujoco_data["mujoco_dof_pos"]
    mujoco_dof_vel = mujoco_data["mujoco_dof_vel"]
    mujoco_gvec = mujoco_data["mujoco_gvec"]

    ref_motion_phase = (counter + 1) * cfg.simulation_dt / cfg.cycle_time
    ref_motion_phase = np.clip(ref_motion_phase,0,1)
    num_obs_input = (cfg.frame_stack+1) * cfg.num_single_obs

    obs_all =  np.zeros([1,  num_obs_input], dtype=np.float32)
    obs_sigle = np.zeros([1, cfg.num_single_obs], dtype=np.float32)
    obs_sigle[0, 0:23] = action
    obs_sigle[0, 23:26] = mujoco_base_angvel * cfg.obs_scale_base_ang_vel
    obs_sigle[0, 26:49] = (mujoco_dof_pos - cfg.default_dof_pos) * cfg.obs_scale_dof_pos
    obs_sigle[0, 49:72] = mujoco_dof_vel  * cfg.obs_scale_dof_vel
    obs_sigle[0, 72:75] = mujoco_gvec * cfg.obs_scale_gvec
    obs_sigle[0, 75] = ref_motion_phase * cfg.obs_scale_refmotion


    obs_all[0,0:23] = obs_sigle[0,0:23].copy()   
    obs_all[0,23:26] = obs_sigle[0,23:26].copy() 
    obs_all[0,26:49] = obs_sigle[0,26:49].copy() 
    obs_all[0,49:72] =  obs_sigle[0,49:72].copy() 
    # 72:164 action;
    # 164:176 base_ang_vel
    # 176:268 dof_pos
    # 268:360 dof_vel
    # 360:372 gravity
    # 372:376 phase
    obs_all[0,72:376] = hist_obs_c[0] * cfg.obs_scale_hist 
    obs_all[0,376:379] = obs_sigle[0,72:75].copy() 
    obs_all[0,379] = obs_sigle[0,75].copy() 

    hist_obs_cat = update_hist_obs(hist_dict,obs_sigle)
    obs_all = np.clip(obs_all, -cfg.clip_observations, cfg.clip_observations)
    
    return obs_all,hist_obs_cat
 

def pd_control(target_pos,dof_pos, target_vel,dof_vel ,cfg):
    torque_out = (target_pos  - dof_pos ) * cfg.kps + (target_vel - dof_vel)* cfg.kds
    return torque_out
 
def run_mujoco(cfg):
    # mujoco接口初始化
    model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.simulation_dt
    data.qpos[-cfg.num_actions:] = cfg.default_dof_pos

    mujoco.mj_step(model, data)

    # mujoco可视化设置
    viewer = mujoco_viewer.MujocoViewer(model,data) 
    viewer.cam.distance=3.0 
    viewer.cam.azimuth = 90
    viewer.cam.elevation=-45
    viewer.cam.lookat[:]=np.array([0.0,-0.25,0.824])

    # 策略模型加载
    onnx_model_path = cfg.policy_path
    policy = onnxruntime.InferenceSession(onnx_model_path)

    # 变量初始化
    target_dof_pos =np.zeros((1,len(cfg.default_dof_pos.copy())))
   
    action = np.zeros(cfg.num_actions, dtype=np.float32)

    hist_dict = {'actions':np.zeros((cfg.frame_stack,cfg.num_actions), dtype=np.double),
                'base_ang_vel':np.zeros((cfg.frame_stack,3), dtype=np.double),
                'dof_pos':np.zeros((cfg.frame_stack,cfg.num_actions), dtype=np.double),
                'dof_vel':np.zeros((cfg.frame_stack,cfg.num_actions), dtype=np.double),
                'projected_gravity':np.zeros((cfg.frame_stack,3), dtype=np.double),
                'ref_motion_phase':np.zeros((cfg.frame_stack,1), dtype=np.double),
                    }
    history_keys = ['actions', 'base_ang_vel', 'dof_pos',
                     'dof_vel', 'projected_gravity', 'ref_motion_phase']
    hist_obs = []
    for key in history_keys:
        hist_obs.append(hist_dict[key].reshape(1,-1))
    hist_obs_c = np.concatenate(hist_obs,axis=1)
    counter = 0
    
    # 渲染频率控制
    render_frequency = 60  # 目标60 FPS
    render_interval = max(1, int(1.0 / (render_frequency * cfg.simulation_dt)))
    
    # 性能监控
    start_time = time.time()
    frame_count = 0
    last_fps_time = start_time

    ## 执行回合
    for _ in range(int( cfg.simulation_duration / cfg.simulation_dt)):
        mujoco_data = get_mujoco_data(data)
 
        tau = pd_control(target_dof_pos, mujoco_data["mujoco_dof_pos"], 
                        np.zeros_like(cfg.kds), mujoco_data["mujoco_dof_vel"], cfg)  
        tau = np.clip(tau, -cfg.tau_limit, cfg.tau_limit) 
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        counter += 1
        ## 控制频率
        if counter % cfg.control_decimation == 0:
            obs_buff,hist_obs_c = get_obs(hist_obs_c,hist_dict,mujoco_data,action,counter,cfg)
            policy_input = {policy.get_inputs()[0].name: obs_buff}
            action = policy.run(["action"], policy_input)[0]
            action = np.clip(action, -cfg.clip_actions, cfg.clip_actions)
            target_dof_pos = action * cfg.action_scale + cfg.default_dof_pos
        
        # 只在特定频率下渲染，提高FPS
        if counter % render_interval == 0:
            viewer.render()
            frame_count += 1
            
            # 每秒更新一次FPS显示
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                print(f"当前FPS: {fps:.1f}")
                frame_count = 0
                last_fps_time = current_time
    viewer.close()


if __name__ == '__main__':
    current_directory = os.getcwd() 
    print("路径：", current_directory)
    config_file = current_directory + "/g1_config/mujoco_config.yaml"
    cfg = read_conf(config_file) 
    run_mujoco(cfg)
    print("-----done------")
   
