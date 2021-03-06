import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from gym import utils
from collections import namedtuple
from gym.envs.mujoco import mujoco_env


Leg = namedtuple("Leg", [
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    "hip_upper",
    "hip_lower",
    "thigh_lower",
    "thigh_upper",
    "ankle_lower",
    "ankle_upper",
    "thigh_size",
    "ankle_size",
])


DEFAULT_DOG = [
    Leg(x=-0.4,
        y=-0.2,
        z=0.0,
        a=0,
        b=0,
        c=0,
        hip_upper=30,
        hip_lower=-30,
        thigh_upper=75,
        thigh_lower=15,
        ankle_upper=-60,
        ankle_lower=-120,
        thigh_size=0.4,
        ankle_size=0.4),
    Leg(x=0.4,
        y=-0.2,
        z=0.0,
        a=0,
        b=0,
        c=0,
        hip_upper=30,
        hip_lower=-30,
        thigh_upper=75,
        thigh_lower=15,
        ankle_upper=-60,
        ankle_lower=-120,
        thigh_size=0.4,
        ankle_size=0.4),
    Leg(x=0.4,
        y=0.2,
        z=0.0,
        a=0,
        b=0,
        c=0,
        hip_upper=30,
        hip_lower=-30,
        thigh_upper=75,
        thigh_lower=15,
        ankle_upper=-60,
        ankle_lower=-120,
        thigh_size=0.4,
        ankle_size=0.4),
    Leg(x=-0.4,
        y=0.2,
        z=0.0,
        a=0,
        b=0,
        c=0,
        hip_upper=30,
        hip_lower=-30,
        thigh_upper=75,
        thigh_lower=15,
        ankle_upper=-60,
        ankle_lower=-120,
        thigh_size=0.4,
        ankle_size=0.4),
]


LOWER_BOUND = Leg(
    x=-0.4,
    y=-0.2,
    z=0,
    a=0,
    b=0,
    c=10,
    hip_upper=20,
    hip_lower=-40,
    thigh_upper=65,
    thigh_lower=5,
    ankle_upper=-70,
    ankle_lower=-130,
    thigh_size=0.2,
    ankle_size=0.2
)


UPPER_BOUND = Leg(
    x=0.4,
    y=0.2,
    z=0,
    a=0,
    b=0,
    c=-10,
    hip_upper=40,
    hip_lower=-20,
    thigh_upper=85,
    thigh_lower=25,
    ankle_upper=-50,
    ankle_lower=-110,
    thigh_size=0.4,
    ankle_size=0.4
)


class MorphingDogEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 legs=DEFAULT_DOG):

        xml_name = 'assets/empty_dog.xml'
        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), xml_name)

        tree = ET.parse(xml_path)
        torso = tree.find(".//body[@name='torso']")
        actuator = tree.find(".//actuator")

        for i, leg in enumerate(legs):

            leg_i_body = ET.SubElement(
                torso,
                "body",
                name=f"leg_{i}_body",
                pos=f"{leg.x} {leg.y} {leg.z}",
                euler=f"{leg.a} {leg.b} {leg.c}"
            )

            ET.SubElement(
                leg_i_body,
                "joint",
                axis="1 0 0",
                name=f"hip_{i}_joint",
                pos="0.0 0.0 0.0",
                range=f"{leg.hip_lower} {leg.hip_upper}",
                type="hinge",
            )

            ET.SubElement(
                leg_i_body,
                "joint",
                axis="0 1 0",
                name=f"thigh_{i}_joint",
                pos="0.0 0.0 0.0",
                range=f"{leg.thigh_lower} {leg.thigh_upper}",
                type="hinge",
            )

            ET.SubElement(
                leg_i_body,
                "geom",
                fromto=f"0.0 0.0 -{leg.thigh_size} 0.0 0.0 0.0",
                name=f"thigh_{i}_geom",
                size="0.08",
                type="capsule",
            )

            ankle_i_body = ET.SubElement(
                leg_i_body,
                "body",
                pos=f"0 0 -{leg.thigh_size}",
                name=f"ankle_{i}_geom",
            )

            ET.SubElement(
                ankle_i_body,
                "joint",
                axis="0 1 0",
                name=f"ankle_{i}_joint",
                pos="0.0 0.0 0.0",
                range=f"{leg.ankle_lower} {leg.ankle_upper}",
                type="hinge",
            )

            ET.SubElement(
                ankle_i_body,
                "geom",
                fromto=f"0.0 0.0 -{leg.ankle_size} 0.0 0.0 0.0",
                name=f"ankle_{i}_geom",
                size="0.08",
                type="capsule",
            )

            ET.SubElement(
                actuator,
                "motor",
                ctrllimited="true",
                ctrlrange="-1.0 1.0",
                joint=f"hip_{i}_joint",
                gear="150",
            )

            ET.SubElement(
                actuator,
                "motor",
                ctrllimited="true",
                ctrlrange="-1.0 1.0",
                joint=f"thigh_{i}_joint",
                gear="150",
            )

            ET.SubElement(
                actuator,
                "motor",
                ctrllimited="true",
                ctrlrange="-1.0 1.0",
                joint=f"ankle_{i}_joint",
                gear="150",
            )

        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)

        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):

        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        forward_reward = (xposafter - xposbefore) / self.dt
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() and 0.2 <= state[2] <= 1.25
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
        qpos = self.init_qpos + \
               self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + \
               self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
