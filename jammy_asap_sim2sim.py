## B站：飞的岛
# 微信：  feidedaoRobot
## refer
##https://github.com/LeCAR-Lab/ASAP
##https://github.com/engineai-robotics/engineai_legged_gym
##https://github.com/unitreerobotics/unitree_rl_gym
 

# 导入MuJoCo物理引擎和可视化工具
import mujoco, mujoco_viewer  # pip install mujoco-python-viewer
# 导入数值计算库
import numpy as np
# 导入ONNX运行时，用于执行预训练的策略模型
import onnxruntime
# 导入YAML配置文件解析库
import yaml
# 导入操作系统接口
import os
# 导入时间模块，用于性能监控
import time
# 导入旋转矩阵处理库，用于坐标系变换
from scipy.spatial.transform import Rotation as R
# 导入简单命名空间，用于配置参数管理
from types import SimpleNamespace
# 导入日志模块
import logging
# 导入JSON模块，用于保存调试数据
import json
# 导入日期时间模块
from datetime import datetime

# 配置日志系统
def setup_logging(log_level=logging.INFO):
    """
    设置日志系统
    """
    # 清除现有的日志配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/simulation_{timestamp}.log'
    
    # 配置日志格式
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ],
        force=True  # 强制重新配置
    )
    
    # 测试日志是否工作
    logging.info(f"日志系统初始化完成，日志文件: {log_filename}")
    print(f"日志系统初始化完成，日志文件: {log_filename}")  # 确保控制台输出
    return log_filename

# 数据监控类
class DataMonitor:
    """
    数据监控类，用于监控和分析仿真数据
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_history = {
            'actions': [],
            'target_positions': [],
            'current_positions': [],
            'current_velocities': [],
            'torques': [],
            'observations': [],
            'base_ang_vel': [],
            'projected_gravity': [],
            'motion_phase': [],
            'timestamps': []
        }
        self.error_count = 0
        self.warning_count = 0
        
    def log_data(self, step, action, target_pos, current_pos, current_vel, torque, obs, mujoco_data, phase):
        """
        记录仿真数据
        """
        timestamp = time.time()
        
        # 记录数据
        self.data_history['actions'].append(action.copy())
        self.data_history['target_positions'].append(target_pos.copy())
        self.data_history['current_positions'].append(current_pos.copy())
        self.data_history['current_velocities'].append(current_vel.copy())
        self.data_history['torques'].append(torque.copy())
        self.data_history['observations'].append(obs.copy())
        self.data_history['base_ang_vel'].append(mujoco_data['mujoco_base_angvel'].copy())
        self.data_history['projected_gravity'].append(mujoco_data['mujoco_gvec'].copy())
        self.data_history['motion_phase'].append(phase)
        self.data_history['timestamps'].append(timestamp)
        
        # 检查数据异常
        self._check_data_anomalies(step, action, target_pos, current_pos, torque, obs)
        
        # 每100步输出一次数据统计
        if step % 100 == 0:
            self._log_statistics(step)
    
    def _check_data_anomalies(self, step, action, target_pos, current_pos, torque, obs):
        """
        检查数据异常
        """
        # 检查动作是否包含NaN或无穷大
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            logging.error(f"步骤 {step}: 动作包含NaN或无穷大值")
            self.error_count += 1
            
        # 检查目标位置是否超出关节限制
        joint_limits_violation = np.any(np.abs(target_pos) > np.pi)
        if joint_limits_violation:
            logging.warning(f"步骤 {step}: 目标位置超出关节限制")
            self.warning_count += 1
            
        # 检查力矩是否过大
        max_torque = np.max(np.abs(torque))
        if max_torque > 100:  # 假设100N⋅m为异常阈值
            logging.warning(f"步骤 {step}: 力矩过大 {max_torque:.2f} N⋅m")
            self.warning_count += 1
            
        # 检查观测数据是否异常
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            logging.error(f"步骤 {step}: 观测数据包含NaN或无穷大值")
            self.error_count += 1
    
    def _log_statistics(self, step):
        """
        输出数据统计信息
        """
        if len(self.data_history['actions']) == 0:
            return
            
        actions = np.array(self.data_history['actions'][-100:])  # 最近100步
        torques = np.array(self.data_history['torques'][-100:])
        target_pos = np.array(self.data_history['target_positions'][-100:])
        current_pos = np.array(self.data_history['current_positions'][-100:])
        
        logging.info(f"步骤 {step} 统计:")
        logging.info(f"  动作范围: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
        logging.info(f"  力矩范围: [{np.min(torques):.3f}, {np.max(torques):.3f}]")
        logging.info(f"  位置误差: {np.mean(np.abs(target_pos - current_pos)):.3f}")
        logging.info(f"  错误计数: {self.error_count}, 警告计数: {self.warning_count}")
    
    def save_debug_data(self, filename):
        """
        保存调试数据到文件
        """
        # 转换为可序列化的格式
        debug_data = {}
        for key, value in self.data_history.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    debug_data[key] = [v.tolist() for v in value]
                else:
                    debug_data[key] = value
        
        with open(filename, 'w') as f:
            json.dump(debug_data, f, indent=2)
        
        logging.info(f"调试数据已保存到: {filename}")

def quaternion_to_euler_array(quat):
    """
    将四元数转换为欧拉角数组
    输入: quat - 四元数 [x, y, z, w]
    输出: 欧拉角数组 [roll, pitch, yaw] (弧度)
    """
    # 确保四元数格式正确 [x, y, z, w]
    x, y, z, w = quat
    
    # 计算Roll角（绕x轴旋转）
    t0 = +2.0 * (w * x + y * z)  # 分子
    t1 = +1.0 - 2.0 * (x * x + y * y)  # 分母
    roll_x = np.arctan2(t0, t1)  # 使用arctan2避免除零错误
    
    # 计算Pitch角（绕y轴旋转）
    t2 = +2.0 * (w * y - z * x)  # 分子
    t2 = np.clip(t2, -1.0, 1.0)  # 限制在[-1,1]范围内，避免数值误差
    pitch_y = np.arcsin(t2)  # 计算pitch角
    
    # 计算Yaw角（绕z轴旋转）
    t3 = +2.0 * (w * z + x * y)  # 分子
    t4 = +1.0 - 2.0 * (y * y + z * z)  # 分母
    yaw_z = np.arctan2(t3, t4)  # 使用arctan2计算yaw角
    
    # 返回roll, pitch, yaw的NumPy数组（弧度制）
    return np.array([roll_x, pitch_y, yaw_z])

def read_conf(config_file):
    """
    读取YAML配置文件并解析为配置对象
    输入: config_file - 配置文件路径
    输出: cfg - 包含所有配置参数的命名空间对象
    """
    # 创建简单命名空间对象来存储配置
    cfg = SimpleNamespace()
    # 打开并读取YAML配置文件
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

     # 解析观测相关配置参数:
    cfg.num_single_obs = config["num_single_obs"]  # 单帧观测维度
    cfg.simulation_dt = config["simulation_dt"]  # 仿真时间步长
    cfg.cycle_time = config["cycle_time"]  # 运动周期时间
    cfg.frame_stack = config["frame_stack"]  # 历史帧堆叠数量


    # 默认关节位置配置
    cfg.default_dof_pos = np.array(config["default_dof_pos"], dtype=np.float32)

    # 观测数据缩放因子配置
    cfg.obs_scale_base_ang_vel = config["obs_scale_base_ang_vel"]  # 基座角速度缩放
    cfg.obs_scale_dof_pos = config["obs_scale_dof_pos"]  # 关节位置缩放
    cfg.obs_scale_dof_vel = config["obs_scale_dof_vel"]  # 关节速度缩放
    cfg.obs_scale_gvec = config["obs_scale_gvec"]  # 重力向量缩放
    cfg.obs_scale_refmotion = config["obs_scale_refmotion"]  # 参考运动相位缩放
    cfg.obs_scale_hist = config["obs_scale_hist"]  # 历史数据缩放
 
    # 观测数据限幅值
    cfg.clip_observations = config["clip_observations"]
 
    
    
    # PD控制器增益配置:
    cfg.kps = np.array(config["kps"], dtype=np.float32)  # 位置控制增益
    cfg.kds = np.array(config["kds"], dtype=np.float32)  # 速度控制增益

    # MuJoCo仿真相关配置:
    cfg.xml_path = config["xml_path"]  # 机器人模型XML文件路径
    cfg.num_actions = config["num_actions"]  # 动作空间维度
    cfg.policy_path = config["policy_path"]  # 策略模型文件路径
    cfg.simulation_duration = config["simulation_duration"]  # 仿真持续时间
    cfg.control_decimation = config["control_decimation"]  # 控制频率降采样
    cfg.clip_actions = config["clip_actions"]  # 动作限幅值
    cfg.action_scale = config["action_scale"]  # 动作缩放因子
    cfg.tau_limit = np.array(config["tau_limit"], dtype=np.float32)  # 力矩限制
 
    return cfg
     
def get_mujoco_data(data):
    """
    从MuJoCo仿真器中提取机器人状态数据
    输入: data - MuJoCo数据对象
    输出: mujoco_data - 包含机器人状态的字典
    """
    mujoco_data={}  # 初始化数据字典
    q = data.qpos.astype(np.double)  # 获取关节位置，转换为双精度
    dq = data.qvel.astype(np.double)  # 获取关节速度，转换为双精度
    quat = np.array([q[4], q[5], q[6], q[3]])  # 提取四元数 [x,y,z,w]格式
    r = R.from_quat(quat)  # 创建旋转对象
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # 将线速度转换到机器人坐标系
    base_angvel = dq[3:6]  # 提取基座角速度
    # line_acc = data.sensor('imu-linear-acceleration').data.astype(np.double)  # 线性加速度（已注释）
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)  # 重力向量在机器人坐标系中的投影
    
    # import math
    # root_euler = quaternion_to_euler_array(quat)  # 转换为欧拉角（已注释）
    # root_euler[root_euler > math.pi] -= 2 * math.pi  # 角度归一化（已注释）

    # 存储提取的数据到字典中
    mujoco_data['mujoco_dof_pos'] = q[7:]  # 关节位置（排除基座）
    mujoco_data['mujoco_dof_vel'] = dq[6:]  # 关节速度（排除基座）
    mujoco_data['mujoco_base_angvel'] = base_angvel  # 基座角速度
    mujoco_data['mujoco_gvec'] = gvec  # 重力向量投影
    return mujoco_data
 
def update_hist_obs(hist_dict, obs_sigle):
    """
    更新历史观测数据，维护滑动窗口
    输入: hist_dict - 历史数据字典, obs_sigle - 当前单帧观测
    输出: hist_obs - 更新后的历史观测向量
    '''
        history_keys = ['actions', 'base_ang_vel', 'dof_pos',
                    dof_vel', 'projected_gravity', 'ref_motion_phase']
    '''
    """
    # 定义观测数据中各部分的切片索引
    slices = {
        'actions': slice(0, 23),  # 动作数据索引
        'base_ang_vel': slice(23, 26),  # 基座角速度索引
        'dof_pos': slice(26, 49),  # 关节位置索引
        'dof_vel': slice(49, 72),  # 关节速度索引
        'projected_gravity': slice(72, 75),  # 重力投影索引
        'ref_motion_phase': slice(75, 76)  # 运动相位索引
    }
    
    # 遍历每个历史数据键
    for key, slc in slices.items():
        # 删除最旧的条目并添加新的观测
        arr = np.delete(hist_dict[key], -1, axis=0)  # 删除最后一行（最旧数据）
        arr = np.vstack((obs_sigle[0, slc], arr))  # 在开头添加新数据
        hist_dict[key] = arr  # 更新历史字典
    
    # 将所有历史数据连接成一个大向量
    hist_obs = np.concatenate([
        hist_dict[key].reshape(1, -1)  # 将每个历史数据展平
        for key in sorted(hist_dict.keys())
    ], axis=1).astype(np.float32)  # 沿列方向连接
    return hist_obs

def get_obs(hist_obs_c,hist_dict,mujoco_data,action,counter,cfg):
    """
    构建完整的观测向量
    输入: hist_obs_c - 历史观测, hist_dict - 历史字典, mujoco_data - 当前状态, action - 当前动作, counter - 计数器, cfg - 配置
    输出: obs_all - 完整观测向量, hist_obs_cat - 更新后的历史观测
    ''' obs:
    action #  23
    base_ang_vel # 3
    dof_pos # 23
    dof_vel # 23
    history_actor # 4 * (23+3+23+23+3+1)=4*76=304
    projected_gravity # 3
    ref_motion_phase # 1 
    '''
    """
    # 从MuJoCo数据中提取各个状态分量
    mujoco_base_angvel = mujoco_data["mujoco_base_angvel"]  # 基座角速度
    mujoco_dof_pos = mujoco_data["mujoco_dof_pos"]  # 关节位置
    mujoco_dof_vel = mujoco_data["mujoco_dof_vel"]  # 关节速度
    mujoco_gvec = mujoco_data["mujoco_gvec"]  # 重力向量

    # 计算参考运动相位（基于时间的周期性）
    ref_motion_phase = (counter + 1) * cfg.simulation_dt / cfg.cycle_time  # 计算当前相位
    ref_motion_phase = np.clip(ref_motion_phase,0,1)  # 限制在[0,1]范围内
    num_obs_input = (cfg.frame_stack+1) * cfg.num_single_obs  # 计算总观测维度

    # 初始化观测数组
    obs_all =  np.zeros([1,  num_obs_input], dtype=np.float32)  # 完整观测向量
    obs_sigle = np.zeros([1, cfg.num_single_obs], dtype=np.float32)  # 单帧观测向量
    
    # 构建单帧观测向量
    obs_sigle[0, 0:23] = action  # 动作数据
    obs_sigle[0, 23:26] = mujoco_base_angvel * cfg.obs_scale_base_ang_vel  # 基座角速度（缩放后）
    obs_sigle[0, 26:49] = (mujoco_dof_pos - cfg.default_dof_pos) * cfg.obs_scale_dof_pos  # 关节位置误差（缩放后）
    obs_sigle[0, 49:72] = mujoco_dof_vel  * cfg.obs_scale_dof_vel  # 关节速度（缩放后）
    obs_sigle[0, 72:75] = mujoco_gvec * cfg.obs_scale_gvec  # 重力向量（缩放后）
    obs_sigle[0, 75] = ref_motion_phase * cfg.obs_scale_refmotion  # 运动相位（缩放后）


    # 将当前观测数据复制到完整观测向量的前半部分
    obs_all[0,0:23] = obs_sigle[0,0:23].copy()   # 当前动作
    obs_all[0,23:26] = obs_sigle[0,23:26].copy()  # 当前基座角速度
    obs_all[0,26:49] = obs_sigle[0,26:49].copy()  # 当前关节位置
    obs_all[0,49:72] =  obs_sigle[0,49:72].copy()  # 当前关节速度
    # 历史数据索引说明:
    # 72:164 action;  # 历史动作
    # 164:176 base_ang_vel  # 历史基座角速度
    # 176:268 dof_pos  # 历史关节位置
    # 268:360 dof_vel  # 历史关节速度
    # 360:372 gravity  # 历史重力向量
    # 372:376 phase  # 历史运动相位
    obs_all[0,72:376] = hist_obs_c[0] * cfg.obs_scale_hist  # 历史观测数据（缩放后）
    obs_all[0,376:379] = obs_sigle[0,72:75].copy()  # 当前重力向量
    obs_all[0,379] = obs_sigle[0,75].copy()  # 当前运动相位

    # 更新历史观测数据
    hist_obs_cat = update_hist_obs(hist_dict,obs_sigle)
    # 对观测数据进行限幅处理
    obs_all = np.clip(obs_all, -cfg.clip_observations, cfg.clip_observations)
    
    return obs_all,hist_obs_cat
 

def pd_control(target_pos,dof_pos, target_vel,dof_vel ,cfg):
    """
    PD控制器，计算关节力矩
    输入: target_pos - 目标位置, dof_pos - 当前位置, target_vel - 目标速度, dof_vel - 当前速度, cfg - 配置
    输出: torque_out - 计算出的力矩
    """
    # PD控制公式: τ = Kp * (目标位置 - 当前位置) + Kd * (目标速度 - 当前速度)
    torque_out = (target_pos  - dof_pos ) * cfg.kps + (target_vel - dof_vel)* cfg.kds
    return torque_out
 
def run_mujoco(cfg):
    """
    主要的MuJoCo仿真运行函数
    输入: cfg - 配置对象
    """
    # 初始化日志系统
    log_filename = setup_logging(logging.INFO)
    logging.info("开始MuJoCo仿真")
    logging.info(f"配置文件参数:")
    logging.info(f"  仿真时间步长: {cfg.simulation_dt}")
    logging.info(f"  控制频率: {1.0/(cfg.simulation_dt * cfg.control_decimation):.1f} Hz")
    logging.info(f"  默认关节位置: {cfg.default_dof_pos}")
    logging.info(f"  动作缩放: {cfg.action_scale}")
    
    # 初始化数据监控器
    monitor = DataMonitor(cfg)
    
    try:
        # MuJoCo接口初始化
        logging.info("正在加载MuJoCo模型...")
        model = mujoco.MjModel.from_xml_path(cfg.xml_path)  # 从XML文件加载机器人模型
        data = mujoco.MjData(model)  # 创建MuJoCo数据对象
        model.opt.timestep = cfg.simulation_dt  # 设置仿真时间步长
        data.qpos[-cfg.num_actions:] = cfg.default_dof_pos  # 设置初始关节位置

        logging.info(f"MuJoCo模型加载成功: {cfg.xml_path}")

        mujoco.mj_step(model, data)  # 执行一步仿真

        # MuJoCo可视化设置
        logging.info("初始化可视化...")
        viewer = mujoco_viewer.MujocoViewer(model,data)  # 创建可视化查看器
        viewer.cam.distance=3.0  # 设置相机距离
        viewer.cam.azimuth = 90  # 设置相机方位角
        viewer.cam.elevation=-45  # 设置相机仰角
        viewer.cam.lookat[:]=np.array([0.0,-0.25,0.824])  # 设置相机观察点

        # 策略模型加载
        logging.info("正在加载策略模型...")
        onnx_model_path = cfg.policy_path  # 获取模型文件路径
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"策略模型文件不存在: {onnx_model_path}")
        
        policy = onnxruntime.InferenceSession(onnx_model_path)  # 创建ONNX推理会话
        logging.info(f"策略模型加载成功: {onnx_model_path}")
        
        # 检查模型输入输出
        input_name = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name
        logging.info(f"模型输入: {input_name}, 输出: {output_name}")

        # 变量初始化
        target_dof_pos =np.zeros((1,len(cfg.default_dof_pos.copy())))  # 目标关节位置
       
        action = np.zeros(cfg.num_actions, dtype=np.float32)  # 动作向量

        # 初始化历史数据字典，用于存储滑动窗口的历史观测
        hist_dict = {'actions':np.zeros((cfg.frame_stack,cfg.num_actions), dtype=np.double),  # 历史动作
                    'base_ang_vel':np.zeros((cfg.frame_stack,3), dtype=np.double),  # 历史基座角速度
                    'dof_pos':np.zeros((cfg.frame_stack,cfg.num_actions), dtype=np.double),  # 历史关节位置
                    'dof_vel':np.zeros((cfg.frame_stack,cfg.num_actions), dtype=np.double),  # 历史关节速度
                    'projected_gravity':np.zeros((cfg.frame_stack,3), dtype=np.double),  # 历史重力向量
                    'ref_motion_phase':np.zeros((cfg.frame_stack,1), dtype=np.double),  # 历史运动相位
                        }
        # 定义历史数据的键名
        history_keys = ['actions', 'base_ang_vel', 'dof_pos',
                         'dof_vel', 'projected_gravity', 'ref_motion_phase']
        # 初始化历史观测向量
        hist_obs = []
        for key in sorted(history_keys):
            print(key)
            hist_obs.append(hist_dict[key].reshape(1,-1))  # 将每个历史数据展平
        hist_obs_c = np.concatenate(hist_obs,axis=1)  # 连接所有历史数据
        counter = 0  # 仿真步计数器
        
        # 渲染频率控制
        render_frequency = 60  # 目标60 FPS
        render_interval = max(1, int(1.0 / (render_frequency * cfg.simulation_dt)))  # 计算渲染间隔
        
        # 性能监控
        start_time = time.time()  # 记录开始时间
        frame_count = 0  # 渲染帧计数器
        last_fps_time = start_time  # 上次FPS计算时间
        
        # 计算总步数
        total_steps = int(cfg.simulation_duration / cfg.simulation_dt)
        logging.info(f"仿真总步数: {total_steps}")
        logging.info(f"仿真开始...")

        ## 执行仿真回合
        for step in range(total_steps):  # 根据仿真时长和时间步长计算总步数
            try:
                mujoco_data = get_mujoco_data(data)  # 获取当前机器人状态数据
     
                # 执行PD控制，计算关节力矩
                tau = pd_control(target_dof_pos, mujoco_data["mujoco_dof_pos"], 
                                np.zeros_like(cfg.kds), mujoco_data["mujoco_dof_vel"], cfg)  # 目标速度设为0
                tau = np.clip(tau, -cfg.tau_limit, cfg.tau_limit)  # 限制力矩在安全范围内
                data.ctrl[:] = tau  # 将力矩应用到机器人
                mujoco.mj_step(model, data)  # 执行一步物理仿真

                counter += 1  # 增加计数器
                ## 控制频率处理
                if counter % cfg.control_decimation == 0:  # 每control_decimation步执行一次策略
                    # 获取观测数据
                    obs_buff,hist_obs_c = get_obs(hist_obs_c,hist_dict,mujoco_data,action,counter,cfg)
                    
                    # 检查观测数据维度
                    expected_obs_dim = (cfg.frame_stack+1) * cfg.num_single_obs
                    if obs_buff.shape[1] != expected_obs_dim:
                        logging.error(f"观测维度不匹配: 期望 {expected_obs_dim}, 实际 {obs_buff.shape[1]}")
                    
                    # 准备策略输入
                    policy_input = {policy.get_inputs()[0].name: obs_buff}  # 设置ONNX模型输入
                    
                    # 执行策略推理
                    try:
                        action = policy.run(["action"], policy_input)[0]  # 运行ONNX模型获取动作
                        action = np.clip(action, -cfg.clip_actions, cfg.clip_actions)  # 限制动作范围
                        
                        # 记录策略推理结果
                        if step % 100 == 0:
                            logging.info(f"步骤 {step}: 策略推理成功，动作范围 [{np.min(action):.3f}, {np.max(action):.3f}]")
                            
                    except Exception as e:
                        logging.error(f"步骤 {step}: 策略推理失败: {e}")
                        # 使用零动作作为后备
                        action = np.zeros(cfg.num_actions, dtype=np.float32)
                    
                    # 计算目标关节位置
                    target_dof_pos = action * cfg.action_scale + cfg.default_dof_pos  # 动作缩放后加上默认位置
                    
                    # 记录数据到监控器
                    phase = (counter + 1) * cfg.simulation_dt / cfg.cycle_time
                    phase = np.clip(phase, 0, 1)
                    monitor.log_data(step, action, target_dof_pos[0], 
                                   mujoco_data["mujoco_dof_pos"], 
                                   mujoco_data["mujoco_dof_vel"], 
                                   tau, obs_buff[0], mujoco_data, phase)
                
                # 只在特定频率下渲染，提高FPS
                if counter % render_interval == 0:  # 按渲染间隔进行渲染
                    viewer.render()  # 渲染当前帧
                    frame_count += 1  # 增加渲染帧计数
                    
                    # 每秒更新一次FPS显示
                    current_time = time.time()  # 获取当前时间
                    if current_time - last_fps_time >= 1.0:  # 如果距离上次FPS计算超过1秒
                        fps = frame_count / (current_time - last_fps_time)  # 计算FPS
                        logging.info(f"当前FPS: {fps:.1f}")  # 打印FPS
                        frame_count = 0  # 重置帧计数
                        last_fps_time = current_time  # 更新上次FPS计算时间
                        
            except Exception as e:
                logging.error(f"步骤 {step} 执行失败: {e}")
                # 保存当前状态用于调试
                debug_filename = f'logs/debug_data_step_{step}.json'
                monitor.save_debug_data(debug_filename)
                raise e
        
        # 仿真完成
        total_time = time.time() - start_time
        logging.info(f"仿真完成，总耗时: {total_time:.2f} 秒")
        logging.info(f"平均FPS: {total_steps / total_time:.1f}")
        
        # 保存最终调试数据
        final_debug_filename = f'logs/final_debug_data.json'
        monitor.save_debug_data(final_debug_filename)
        
        viewer.close()  # 关闭可视化窗口
        
    except Exception as e:
        logging.error(f"仿真过程中发生错误: {e}")
        # 保存错误时的调试数据
        error_debug_filename = f'logs/error_debug_data.json'
        monitor.save_debug_data(error_debug_filename)
        raise e

def analyze_joint_behavior(debug_data_file):
    """
    分析关节行为，帮助定位动作问题
    
    Args:
        debug_data_file: 调试数据文件路径
    """
    try:
        with open(debug_data_file, 'r') as f:
            data = json.load(f)
        
        logging.info("=== 关节行为分析 ===")
        
        # 分析动作变化
        actions = np.array(data['actions'])
        target_positions = np.array(data['target_positions'])
        current_positions = np.array(data['current_positions'])
        
        # 计算动作统计
        action_mean = np.mean(actions, axis=0)
        action_std = np.std(actions, axis=0)
        action_range = np.max(actions, axis=0) - np.min(actions, axis=0)
        
        logging.info("动作统计:")
        for i in range(len(action_mean)):
            logging.info(f"  关节 {i}: 均值={action_mean[i]:.3f}, 标准差={action_std[i]:.3f}, 范围={action_range[i]:.3f}")
        
        # 分析位置跟踪误差
        position_errors = np.abs(target_positions - current_positions)
        mean_errors = np.mean(position_errors, axis=0)
        max_errors = np.max(position_errors, axis=0)
        
        logging.info("位置跟踪误差:")
        for i in range(len(mean_errors)):
            logging.info(f"  关节 {i}: 平均误差={mean_errors[i]:.3f}, 最大误差={max_errors[i]:.3f}")
        
        # 分析力矩
        torques = np.array(data['torques'])
        torque_mean = np.mean(torques, axis=0)
        torque_max = np.max(np.abs(torques), axis=0)
        
        logging.info("力矩统计:")
        for i in range(len(torque_mean)):
            logging.info(f"  关节 {i}: 平均力矩={torque_mean[i]:.3f}, 最大力矩={torque_max[i]:.3f}")
        
        # 检测异常关节
        problematic_joints = []
        for i in range(len(action_mean)):
            if action_std[i] > 0.5 or max_errors[i] > 0.1 or torque_max[i] > 50:
                problematic_joints.append(i)
        
        if problematic_joints:
            logging.warning(f"检测到异常关节: {problematic_joints}")
            
            # 详细分析异常关节
            for joint_id in problematic_joints:
                logging.warning(f"关节 {joint_id} 详细分析:")
                logging.warning(f"  动作标准差: {action_std[joint_id]:.3f}")
                logging.warning(f"  最大位置误差: {max_errors[joint_id]:.3f}")
                logging.warning(f"  最大力矩: {torque_max[joint_id]:.3f}")
        else:
            logging.info("所有关节行为正常")
            
    except Exception as e:
        logging.error(f"分析关节行为时出错: {e}")
        import traceback
        logging.error(f"错误详情: {traceback.format_exc()}")

def check_configuration_consistency(cfg):
    """
    检查配置参数的一致性
    
    Args:
        cfg: 配置对象
    """
    logging.info("=== 配置一致性检查 ===")
    
    # 检查数组长度一致性
    expected_length = cfg.num_actions
    
    arrays_to_check = [
        ('default_dof_pos', cfg.default_dof_pos),
        ('kps', cfg.kps),
        ('kds', cfg.kds),
        ('tau_limit', cfg.tau_limit)
    ]
    
    for name, array in arrays_to_check:
        if len(array) != expected_length:
            logging.error(f"{name} 长度不匹配: 期望 {expected_length}, 实际 {len(array)}")
        else:
            logging.info(f"{name} 长度正确: {len(array)}")
    
    # 检查观测维度
    expected_obs_dim = (cfg.frame_stack + 1) * cfg.num_single_obs
    logging.info(f"期望观测维度: {expected_obs_dim}")
    
    # 检查时间参数
    if cfg.simulation_dt <= 0:
        logging.error("仿真时间步长必须大于0")
    
    if cfg.control_decimation <= 0:
        logging.error("控制频率降采样必须大于0")
    
    control_freq = 1.0 / (cfg.simulation_dt * cfg.control_decimation)
    logging.info(f"实际控制频率: {control_freq:.1f} Hz")

def validate_model_inputs(policy, cfg):
    """
    验证模型输入的正确性
    
    Args:
        policy: ONNX模型
        cfg: 配置对象
    """
    logging.info("=== 模型输入验证 ===")
    
    # 获取模型输入信息
    inputs = policy.get_inputs()
    outputs = policy.get_outputs()
    
    logging.info(f"模型输入数量: {len(inputs)}")
    logging.info(f"模型输出数量: {len(outputs)}")
    
    for i, input_info in enumerate(inputs):
        logging.info(f"输入 {i}: {input_info.name}, 形状: {input_info.shape}, 类型: {input_info.type}")
    
    for i, output_info in enumerate(outputs):
        logging.info(f"输出 {i}: {output_info.name}, 形状: {output_info.shape}, 类型: {output_info.type}")
    
    # 检查输入维度
    expected_obs_dim = (cfg.frame_stack + 1) * cfg.num_single_obs
    if inputs[0].shape[1] != expected_obs_dim:
        logging.error(f"模型输入维度不匹配: 期望 {expected_obs_dim}, 实际 {inputs[0].shape[1]}")
    else:
        logging.info("模型输入维度正确")

def test_logging_system():
    """
    测试日志系统是否正常工作
    """
    print("测试日志系统...")
    
    # 测试不同级别的日志
    logging.debug("这是一条DEBUG日志")
    logging.info("这是一条INFO日志")
    logging.warning("这是一条WARNING日志")
    logging.error("这是一条ERROR日志")
    
    # 测试中文日志
    logging.info("测试中文日志: 机器人仿真系统")
    
    # 测试数值日志
    test_value = 123.456
    logging.info(f"测试数值日志: {test_value}")
    
    print("日志系统测试完成")

def analyze_torque_issues(debug_data_file, cfg):
    """
    分析力矩问题并提供解决建议
    
    Args:
        debug_data_file: 调试数据文件路径
        cfg: 配置对象
    """
    try:
        with open(debug_data_file, 'r') as f:
            data = json.load(f)
        
        logging.info("=== 力矩问题分析 ===")
        
        torques = np.array(data['torques'])
        actions = np.array(data['actions'])
        
        # 分析每个关节的力矩
        for i in range(torques.shape[1]):
            joint_torques = torques[:, i]
            joint_actions = actions[:, i]
            
            max_torque = np.max(np.abs(joint_torques))
            mean_torque = np.mean(np.abs(joint_torques))
            torque_limit = cfg.tau_limit[i]
            
            logging.info(f"关节 {i} 力矩分析:")
            logging.info(f"  最大力矩: {max_torque:.2f} N⋅m (限制: {torque_limit:.2f} N⋅m)")
            logging.info(f"  平均力矩: {mean_torque:.2f} N⋅m")
            logging.info(f"  利用率: {max_torque/torque_limit*100:.1f}%")
            
            if max_torque > torque_limit * 0.8:
                logging.warning(f"  关节 {i} 力矩接近限制，建议调整PD参数")
                
                # 提供调整建议
                current_kp = cfg.kps[i]
                current_kd = cfg.kds[i]
                
                logging.info(f"  当前PD参数: Kp={current_kp}, Kd={current_kd}")
                logging.info(f"  建议调整: Kp={current_kp*0.8:.1f}, Kd={current_kd*1.2:.1f}")
        
        # 整体建议
        logging.info("=== 整体建议 ===")
        logging.info("1. 降低PD控制器的刚度参数(Kp)以减少力矩")
        logging.info("2. 增加阻尼参数(Kd)以提高稳定性")
        logging.info("3. 检查默认关节位置是否合理")
        logging.info("4. 考虑调整动作缩放因子")
        
    except Exception as e:
        logging.error(f"分析力矩问题时出错: {e}")
        import traceback
        logging.error(f"错误详情: {traceback.format_exc()}")

if __name__ == '__main__':
    # 主程序入口
    print("程序开始执行...")
    
    # 首先初始化日志系统
    try:
        log_filename = setup_logging(logging.INFO)
        logging.info("程序开始执行")
        print("日志系统初始化成功")
        
        # 测试日志系统
        test_logging_system()
        
    except Exception as e:
        print(f"日志系统初始化失败: {e}")
        # 如果日志系统失败，至少确保程序能继续运行
    
    current_directory = os.getcwd()  # 获取当前工作目录
    print("路径：", current_directory)  # 打印当前路径
    logging.info(f"当前工作目录: {current_directory}")
    
    config_file = current_directory + "/g1_config/mujoco_config.yaml"  # 构建配置文件路径
    logging.info(f"配置文件路径: {config_file}")
    
    try:
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            error_msg = f"配置文件不存在: {config_file}"
            print(error_msg)
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logging.info("开始读取配置文件...")
        cfg = read_conf(config_file)  # 读取配置文件
        logging.info("配置文件读取成功")
        
        # 检查配置一致性
        logging.info("开始检查配置一致性...")
        check_configuration_consistency(cfg)
        
        # 验证模型（如果存在）
        if os.path.exists(cfg.policy_path):
            logging.info(f"开始验证模型: {cfg.policy_path}")
            policy = onnxruntime.InferenceSession(cfg.policy_path)
            validate_model_inputs(policy, cfg)
        else:
            logging.warning(f"策略模型文件不存在: {cfg.policy_path}")
        
        logging.info("开始运行MuJoCo仿真...")
        run_mujoco(cfg)  # 运行MuJoCo仿真
        print("-----done------")  # 打印完成信息
        logging.info("仿真完成")
        
        # 分析调试数据（如果存在）
        debug_files = ['logs/final_debug_data.json', 'logs/error_debug_data.json']
        for debug_file in debug_files:
            if os.path.exists(debug_file):
                logging.info(f"开始分析调试数据: {debug_file}")
                analyze_joint_behavior(debug_file)
                analyze_torque_issues(debug_file, cfg)
                break
                
    except FileNotFoundError as e:
        error_msg = f"文件不存在: {e}"
        print(error_msg)
        logging.error(error_msg)
    except Exception as e:
        error_msg = f"程序执行失败: {e}"
        print(error_msg)
        logging.error(error_msg)
        # 分析错误调试数据（如果存在）
        if os.path.exists('logs/error_debug_data.json'):
            logging.info("开始分析错误调试数据")
            analyze_joint_behavior('logs/error_debug_data.json')
            analyze_torque_issues('logs/error_debug_data.json', cfg)
    
    logging.info("程序结束")
    print("程序结束")
   
