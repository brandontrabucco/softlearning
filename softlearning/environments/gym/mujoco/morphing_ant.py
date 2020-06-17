import numpy as np
import os
import tempfile
from gym import utils
from gym.envs.mujoco import mujoco_env
from itertools import product


class MorphingAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 thigh_size1=0.2,
                 ankle_size1=0.4,
                 thigh_size2=0.2,
                 ankle_size2=0.4,
                 thigh_size3=0.2,
                 ankle_size3=0.4,
                 thigh_size4=0.2,
                 ankle_size4=0.4):

        xml_name = 'assets/ant.xml'
        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), xml_name)
        with open(xml_path, 'r') as f:
            contents = f.read().format(thigh_size1=thigh_size1,
                                       ankle_size1=ankle_size1,
                                       thigh_size2=thigh_size2,
                                       ankle_size2=ankle_size2,
                                       thigh_size3=thigh_size3,
                                       ankle_size3=ankle_size3,
                                       thigh_size4=thigh_size4,
                                       ankle_size4=ankle_size4)

        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        with open(file_path, 'w') as f:
            f.write(contents)
        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() and 0.2 <= state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
