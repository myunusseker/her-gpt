import time
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


class CubeEnvironment:
    """Simple environment where a cube is pushed towards a goal."""

    def __init__(self, gui: bool = True, speed: int = 60, goal_position: list = [0.5, 0.5, 0]):
        self.speed = speed
        self.gui = gui
        self.goal_position = np.array(goal_position)
        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane_id = p.loadURDF("plane.urdf")
        self.block_id = None

        self.last_camera_target = None
        self.last_camera_distance = 0.7 #0.7
        self.last_camera_yaw = -45
        self.last_camera_pitch = -45

    def reset(self, start_pos = [0, 0, 0.02]) -> np.ndarray:
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")

        start_pos = start_pos
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.block_id = p.loadURDF("cube_small.urdf", start_pos, start_orientation)
        p.changeVisualShape(self.block_id, -1, rgbaColor=[0, 0, 1, 1])

        goal_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.1,
            length=0.001,
            rgbaColor=[1, 0, 0, 0.6],
            visualFramePosition=[0, 0, 0],
        )

        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=goal_visual,
            basePosition=self.goal_position,
        )

        return self.get_state()

    def get_state(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.block_id)
        return np.array(pos)

    def apply_action(self, force_vector):
        p.applyExternalForce(
            objectUniqueId=self.block_id,
            linkIndex=-1,
            forceObj=force_vector,
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
        )

        for _ in range(240):
            pos, _ = p.getBasePositionAndOrientation(self.block_id)
            self.last_camera_target = pos
            p.resetDebugVisualizerCamera(
                cameraDistance=self.last_camera_distance,
                cameraYaw=self.last_camera_yaw,
                cameraPitch=self.last_camera_pitch,
                cameraTargetPosition=pos,
            )
            p.stepSimulation()

            if self.gui:
                time.sleep(1 / self.speed)

        return self.get_state()
    
    def safe_image_from_pybullet(self, raw_img, width, height):
        """Converts raw PyBullet RGB image to a valid Image object."""
        if not isinstance(raw_img, np.ndarray):
            raw_img = np.array(raw_img, dtype=np.uint8)

        img = raw_img.reshape((height, width, 4))  # PyBullet always returns RGBA

        try:
            return Image.fromarray(img)  # Let PIL auto-detect RGBA
        except ValueError:
            # Fallback: explicitly set mode
            return Image.fromarray(img, mode="RGBA")

    def render_views(self, topdown_path="topdown.png", side_path="side.png", use_static_side=True):
        # Get block position for tracker view
        block_pos, _ = p.getBasePositionAndOrientation(self.block_id)

        # ---- Top-down camera ----
        view_matrix_top = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0.5, 1],
            cameraTargetPosition=[0.5, 0.5, 0],
            cameraUpVector=[0, 1, 0]
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=3.1
        )

        _, _, rgb_top, _, _ = p.getCameraImage(
            width=512,
            height=512,
            viewMatrix=view_matrix_top,
            projectionMatrix=proj_matrix
        )

        # ---- Side camera ----
        if use_static_side:
            # Static side view from initial position
            view_matrix_side = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.2, 0.2, 0],  # Always look at center
                distance=self.last_camera_distance,
                yaw=self.last_camera_yaw,
                pitch=self.last_camera_pitch,
                roll=0,
                upAxisIndex=2
            )
        else:
            # Tracker camera (match GUI debug camera) - follows the cube
            if self.last_camera_target is None:
                self.last_camera_target = block_pos

            view_matrix_side = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.last_camera_target,
                distance=self.last_camera_distance,
                yaw=self.last_camera_yaw,
                pitch=self.last_camera_pitch,
                roll=0,
                upAxisIndex=2
            )

        _, _, rgb_side, _, _ = p.getCameraImage(
            width=512,
            height=512,
            viewMatrix=view_matrix_side,
            projectionMatrix=proj_matrix,
        )

        img_top = self.safe_image_from_pybullet(rgb_top, width=512, height=512)
        img_side = self.safe_image_from_pybullet(rgb_side, width=512, height=512)

        img_top.save(topdown_path)
        img_side.save(side_path)


    def close(self) -> None:
        p.disconnect()


if __name__ == "__main__":
    env = CubeEnvironment(gui=True, goal_position=[0.3, 0.65, 0])
    env.reset(start_pos=[0.3, 0.65, 0.02])
    #env.apply_action(np.array([60, 60, 0]))
    env.render_views(
        topdown_path="data/test/topdown.png",
        side_path="data/test/side.png",
        use_static_side=True
    )
    exit(0)
    env.reset(start_pos=[0.3, 0.2, 0.02])
    #env.apply_action(np.array([60, 60, 0]))
    env.render_views(
        topdown_path="data/test/topdown2.png",
        side_path="data/test/side2.png",
        use_static_side=True
    )
    env.close()