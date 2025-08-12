import pybullet as p
import pybullet_data
import time
import numpy as np


class PegInsertionEnvironment:
    def __init__(self, gui=True, hz=60, initial_position=[0.40, 0.00], force_consistent_rendering=True):
        self.hz = hz
        self.gui = gui
        self.initial_position = initial_position
        self.force_consistent_rendering = force_consistent_rendering

        # Connect to simulation with rendering setup based on user preference
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Store connection info for debugging
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load ground and table
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0])
        self.table_height = 0.62

        # Load Franka Panda
        start_pos = [0, 0, self.table_height]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.franka_id = p.loadURDF("franka_panda/panda.urdf", start_pos, start_ori, useFixedBase=True)

        # Joint indices for the 7 DOF arm (not including fingers)
        self.joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_indices = [9, 10]  # Finger joint indices for Franka Panda
        self.eef_index = 11  # End effector link index

        # Default initial joint positions
        initial_joint_positions = [0, -0.4, 0, -2.4, 0, 2.0, 0.8]

        # Move to initial joint positions
        for i, pos in zip(self.joint_indices, initial_joint_positions):
            p.resetJointState(self.franka_id, i, pos)

        # Open the gripper
        gripper_open_positions = [0.04, 0.04]
        for i, pos in zip(self.gripper_indices, gripper_open_positions):
            p.resetJointState(self.franka_id, i, pos)

        # Get current EEF pose
        eef_state = p.getLinkState(self.franka_id, self.eef_index)
        self.eef_pos = np.array(eef_state[4])  # position
        self.eef_ori = eef_state[5]            # orientation (quaternion)

        # Create and attach peg
        self._create_and_attach_peg()

        # Create insertion hole on table
        self._create_insertion_hole()

        # Close the gripper to hold the peg
        self._close_gripper()

        print("Peg insertion environment initialized successfully!")

    def _create_and_attach_peg(self):
        """Create the peg and attach it to the end effector."""
        # Peg dimensions (rectangular prism)
        self.peg_width = 0.024   # 2.4cm width (x-axis)
        self.peg_depth = 0.024   # 1.2cm depth (y-axis)
        self.peg_height = 0.08   # 8cm height (z-axis)

        # Create peg collision and visual shapes
        peg_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.peg_width/2, self.peg_depth/2, self.peg_height/2]
        )
        peg_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.peg_width/2, self.peg_depth/2, self.peg_height/2],
            rgbaColor=[0, 1, 0, 1]  # Green peg
        )

        # Calculate proper peg position
        peg_start_pos = [self.eef_pos[0], self.eef_pos[1], self.eef_pos[2] - self.peg_height / 2 + 0.02]

        # Create peg body
        self.peg_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=peg_collision,
            baseVisualShapeIndex=peg_visual,
            basePosition=peg_start_pos,
            baseOrientation=self.eef_ori
        )

        # Create constraint to attach peg to end effector
        self.peg_joint = p.createConstraint(
            parentBodyUniqueId=self.franka_id,
            parentLinkIndex=self.eef_index,
            childBodyUniqueId=self.peg_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, self.peg_height/2 - 0.02],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1]
        )

        # Configure constraint for maximum rigidity
        p.changeConstraint(self.peg_joint, maxForce=50000)

        print(f"Rectangular peg properly attached as fixed joint at position: {peg_start_pos}")

    def _create_insertion_hole(self):
        """Create a hollow box (hole) on the table for peg insertion."""
        # Hole dimensions - slightly larger than peg
        clearance = 0.002  # 2mm clearance on each side
        hole_width = self.peg_width + 2 * clearance   # Slightly wider than peg
        hole_depth = self.peg_depth + 2 * clearance   # Slightly deeper than peg
        hole_height = 0.05  # 5cm deep hole
        wall_thickness = 0.02  # 2cm thick walls

        # Position the hole on the table (you can adjust this position)
        hole_pos = [self.initial_position[0], self.initial_position[1], self.table_height + hole_height/2]

        # Create the outer box (walls)
        outer_width = hole_width + 2 * wall_thickness
        outer_depth = hole_depth + 2 * wall_thickness
        
        outer_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[outer_width/2, outer_depth/2, hole_height/2]
        )
        outer_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[outer_width/2, outer_depth/2, hole_height/2],
            rgbaColor=[0.8, 0.8, 0.8, 1.0]  # Light gray
        )

        # Create the inner box (hole - this will be removed via compound shape)
        inner_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[hole_width/2, hole_depth/2, hole_height/2 + 0.001]  # Slightly taller to ensure clean cut
        )

        # Create compound shape: outer box minus inner box
        # We'll create 4 walls instead of using compound shapes for better physics
        wall_positions = [
            # Front wall
            [hole_pos[0] + (hole_width/2 + wall_thickness/2), hole_pos[1], hole_pos[2]],
            # Back wall  
            [hole_pos[0] - (hole_width/2 + wall_thickness/2), hole_pos[1], hole_pos[2]],
            # Left wall
            [hole_pos[0], hole_pos[1] + (hole_depth/2 + wall_thickness/2), hole_pos[2]],
            # Right wall
            [hole_pos[0], hole_pos[1] - (hole_depth/2 + wall_thickness/2), hole_pos[2]]
        ]
        
        wall_shapes = [
            # Front and back walls (thin in x, full in y)
            [wall_thickness/2, outer_depth/2, hole_height/2],
            [wall_thickness/2, outer_depth/2, hole_height/2],
            # Left and right walls (full in x, thin in y) 
            [hole_width/2, wall_thickness/2, hole_height/2],
            [hole_width/2, wall_thickness/2, hole_height/2]
        ]

        self.hole_walls = []
        for i, (pos, shape) in enumerate(zip(wall_positions, wall_shapes)):
            wall_collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=shape
            )
            wall_visual = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=shape,
                rgbaColor=[1.0, 0.0, 0.0, 1.0]  # Red color
            )
            
            wall_id = p.createMultiBody(
                baseMass=0,  # Static walls
                baseCollisionShapeIndex=wall_collision,
                baseVisualShapeIndex=wall_visual,
                basePosition=pos
            )
            self.hole_walls.append(wall_id)

        # Store hole information for reference
        self.hole_position = hole_pos
        self.hole_width = hole_width
        self.hole_depth = hole_depth
        self.hole_height = hole_height

    def _close_gripper(self):
        """Close the gripper to hold the peg."""
        gripper_closed_positions = [0.01, 0.01]
        for i, pos in zip(self.gripper_indices, gripper_closed_positions):
            p.setJointMotorControl2(
                self.franka_id,
                i,
                p.POSITION_CONTROL,
                pos,
                force=10,
                positionGain=0.3,
                velocityGain=1.0
            )

        # Let the gripper settle
        for _ in range(10):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/self.hz)

    def move_smooth(self, target, duration=3.0, relative=True, stop_on_contact=False, contact_threshold=0.5):
        current_eef_state = p.getLinkState(self.franka_id, self.eef_index)
        start_pos = np.array(current_eef_state[4])
        
        if relative:
            target_pos = start_pos + np.array(target)
        else:
            target_pos = np.array(target)

        total_steps = int(duration * self.hz)        
        for step in range(total_steps):
            if stop_on_contact:
                total_force = 0
                
                contact_points = p.getContactPoints(bodyA=self.peg_id, bodyB=self.table_id)
                for contact in contact_points:
                    normal_force = contact[9]  # Normal force magnitude
                    total_force += abs(normal_force)
                
                for wall_id in self.hole_walls:
                    wall_contacts = p.getContactPoints(bodyA=self.peg_id, bodyB=wall_id)
                    for contact in wall_contacts:
                        normal_force = contact[9]  # Normal force magnitude
                        total_force += abs(normal_force)
                
                if total_force > contact_threshold:
                    print(f"Contact detected! Total force: {total_force:.2f}N - Stopping movement")
                    break
            
            t = step / (total_steps - 1) if total_steps > 1 else 1.0
            
            alpha = 3 * t**2 - 2 * t**3
            
            current_target = start_pos + alpha * (target_pos - start_pos)
            
            joint_positions = p.calculateInverseKinematics(
                self.franka_id, 
                self.eef_index, 
                current_target, 
                self.eef_ori,
                maxNumIterations=100,
                residualThreshold=1e-5
            )

            for j, joint_pos in zip(self.joint_indices, joint_positions):
                p.setJointMotorControl2(
                    self.franka_id, 
                    j, 
                    p.POSITION_CONTROL, 
                    joint_pos, 
                    force=200,
                    positionGain=0.1,
                    velocityGain=1.0
                )

            # Step simulation
            p.stepSimulation()
            if self.gui:
                time.sleep(1/self.hz)

        # Update stored end effector position after movement
        final_eef_state = p.getLinkState(self.franka_id, self.eef_index)
        self.eef_pos = np.array(final_eef_state[4])
        self.eef_ori = final_eef_state[5]

    def get_current_position(self):
        eef_state = p.getLinkState(self.franka_id, self.eef_index)
        current_pos = np.array(eef_state[4])
        current_ori = eef_state[5]
        
        self.eef_pos = current_pos
        self.eef_ori = current_ori
        
        return current_pos

    def render_views(self, save_images=True, side_path="data/peg_insertion_test/side_view.png", wrist_path="data/peg_insertion_test/wrist_view.png"):
        # Ensure simulation is synchronized and OpenGL state is consistent
        p.stepSimulation()
        
        # Get current end effector state for wrist camera
        eef_state = p.getLinkState(self.franka_id, self.eef_index)
        eef_pos = np.array(eef_state[4])
        eef_ori = eef_state[5]  # quaternion
        
        # Convert quaternion to rotation matrix for camera orientation
        eef_rot_matrix = p.getMatrixFromQuaternion(eef_ori)
        eef_rot_matrix = np.array(eef_rot_matrix).reshape(3, 3)
        
        # Wrist camera setup (looking down from end effector, rotated 90 degrees)
        wrist_cam_pos = eef_pos + np.array([-0.1, 0., 0.1])  # 10cm above end effector
        wrist_target = eef_pos + np.array([0, 0, -0.1])   # Looking down
        wrist_up = [1, 0, 0]  # Rotated 90 degrees from [0, 1, 0] to [1, 0, 0]
        
        # Diagonal side camera setup
        side_cam_pos = np.array([0.5, 0.1, self.table_height + 0.2])  # Positioned diagonally
        side_target = np.array([0.4, 0.0, self.table_height])  # Looking at hole position
        side_up = [0, 0, 1]  # Up direction
        
        # Use same simple camera parameters as cube environment
        width, height = 512, 512  
        fov = 60  # Standard FOV like cube environment
        aspect = width / height
        near = 0.01
        far = 10.0
        
        # Compute view and projection matrices for wrist camera
        wrist_view_matrix = p.computeViewMatrix(
            cameraEyePosition=wrist_cam_pos,
            cameraTargetPosition=wrist_target,
            cameraUpVector=wrist_up
        )
        
        # Compute view and projection matrices for side camera
        side_view_matrix = p.computeViewMatrix(
            cameraEyePosition=side_cam_pos,
            cameraTargetPosition=side_target,
            cameraUpVector=side_up
        )
        
        # Projection matrix (same for both cameras)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )
        
        # Use simple rendering like cube environment (no special renderer or flags)
        wrist_img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=wrist_view_matrix,
            projectionMatrix=projection_matrix
            # No renderer or flags - use defaults like cube environment
        )
        
        # Render side camera view with simple rendering
        side_img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=side_view_matrix,
            projectionMatrix=projection_matrix
            # No renderer or flags - use defaults like cube environment
        )
        
        # Extract RGB data like cube environment (keep RGBA format)
        wrist_rgb = np.array(wrist_img[2], dtype=np.uint8).reshape(height, width, 4)[:,:,:3]
        side_rgb = np.array(side_img[2], dtype=np.uint8).reshape(height, width, 4)[:,:,:3]
        print("Wrist RGB shape:", wrist_rgb.shape)
        print(wrist_rgb[0,0])
        # Save images directly like cube environment (no post-processing)
        if save_images:
            # Ensure directories exist
            import os
            os.makedirs(os.path.dirname(wrist_path), exist_ok=True)
            os.makedirs(os.path.dirname(side_path), exist_ok=True)
            
            # Save directly with PIL like cube environment
            from PIL import Image
            Image.fromarray(wrist_rgb).save(wrist_path)
            Image.fromarray(side_rgb).save(side_path)

        return wrist_rgb, side_rgb

    def reset(self):
        """Reset the environment to initial state."""
        # Reset robot to initial joint positions
        initial_joint_positions = [0, -0.4, 0, -2.4, 0, 2.0, 0.8]
        
        # First, reset joint states
        for i, pos in zip(self.joint_indices, initial_joint_positions):
            p.resetJointState(self.franka_id, i, pos)

        # Reset gripper to open position
        gripper_open_positions = [0.04, 0.04]
        for i, pos in zip(self.gripper_indices, gripper_open_positions):
            p.resetJointState(self.franka_id, i, pos)


        # Apply joint motor control to maintain reset positions
        for j, joint_pos in zip(self.joint_indices, initial_joint_positions):
            p.setJointMotorControl2(
                self.franka_id, 
                j, 
                p.POSITION_CONTROL, 
                joint_pos, 
                force=200,
                positionGain=0.3,
                velocityGain=1.0
            )
        
        # Apply gripper motor control
        for i, pos in zip(self.gripper_indices, gripper_open_positions):
            p.setJointMotorControl2(
                self.franka_id,
                i,
                p.POSITION_CONTROL,
                pos,
                force=10,
                positionGain=0.3,
                velocityGain=1.0
            )
        # Apply motor control to ensure robot actually moves to reset positions
        for _ in range(1):  # Give more time for the robot to settle
            p.stepSimulation()
            if self.gui:
                time.sleep(1/self.hz)

        # Update current EEF pose after robot has actually moved
        eef_state = p.getLinkState(self.franka_id, self.eef_index)
        self.eef_pos = np.array(eef_state[4])  # position
        self.eef_ori = eef_state[5]            # orientation (quaternion)

        # Reset peg position to be attached to end effector
        if hasattr(self, 'peg_id'):
            peg_start_pos = [self.eef_pos[0], self.eef_pos[1], self.eef_pos[2] - self.peg_height / 2 + 0.02]
            p.resetBasePositionAndOrientation(self.peg_id, peg_start_pos, self.eef_ori)

        # Close the gripper to hold the peg
        self._close_gripper()

        # Let the simulation settle more
        for _ in range(50):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/self.hz)

        self.move_smooth(target=np.array([self.initial_position[0], self.initial_position[1], self.table_height+0.12]), duration=3.0, relative=False)

        # Final position update to ensure accuracy
        final_eef_state = p.getLinkState(self.franka_id, self.eef_index)
        self.eef_pos = np.array(final_eef_state[4])
        self.eef_ori = final_eef_state[5]

        print("Environment reset to initial state")
        return self.eef_pos

    def disconnect(self):
        """Disconnect from the simulation."""
        p.disconnect()
    
    def close(self):
        """Close the environment (alias for disconnect)."""
        self.disconnect()

    def apply_action(self, offset, initial_env=False, duration=3.0):
        if initial_env:
            self.move_smooth(target=np.array([0.0, 0.0, 0.04]), duration=duration, relative=True)
        else:
            self.move_smooth(target=np.array([offset[0], offset[1], offset[2]]), duration=1.0, relative=True)
            self.move_smooth(
                target=np.array([0.0, 0.0, -0.08]), 
                duration=3.0, 
                relative=True, 
                stop_on_contact=True, 
                contact_threshold=1.0
            )

# Example usage
if __name__ == "__main__":
    # Create environment
    env = PegInsertionEnvironment(gui=False, hz=60)  
    env.reset()
    env.apply_action(offset=np.array([0.0, 0.0, 0.0]))    
    # Render camera views after movement
    print("Rendering camera views...")
    wrist_rgb, side_rgb = env.render_views(save_images=True)
    # Disconnect
    env.disconnect()
