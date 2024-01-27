#!/usr/bin/env python3

from copy import copy
from threading import Thread

import ikfastpy
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial.transform import Rotation


PANDA_JOINT_NAMES = [
	"panda_joint1",
	"panda_joint2",
	"panda_joint3",
	"panda_joint4",
	"panda_joint5",
	"panda_joint6",
	"panda_joint7",
]
JOINT_MINS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_MAXS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

class FrankaRobot(Node):
	def __init__(self, joint_cmd_topic, joint_states_topic):
		"""Interface for controlling franka arm

		:param joint_cmd_topic: The action topic for executing
		joint trajectories

		:param joint_states_topic: Topic where joint state is
		published

		"""

		super().__init__("franka_robot")
		# Action client executing joint trajectories
		# Corresponding action msgs:
		# https://docs.ros.org/en/noetic/api/control_msgs/html/action/FollowJointTrajectory.html
		self.action_client = ActionClient(self, FollowJointTrajectory, joint_cmd_topic)

		# Subscribes to joint states
		self.js_sub = self.create_subscription(
			JointState, joint_states_topic, self.js_cb, 10
		)

		# Provides inverse and forward kinematics
		self.franka_kin = ikfastpy.PyKinematics()

		self.panda_joint_idxs = None
		self.cur_joints = None
		self.move_accepted = None
		self.move_result = None
		self.js_sub

	def js_cb(self, msg):
		"""Cache the latest joint states

		:param msg: the most recent joint states

		"""
		# Map the joint values in the joint states topic to
		# the expected panda joint values
		if self.panda_joint_idxs is None:
			self.panda_joint_idxs = []
			for panda_name in PANDA_JOINT_NAMES:
				for robot_idx, robot_name in enumerate(msg.name):
					if panda_name == robot_name:
						self.panda_joint_idxs.append(robot_idx)
						break
			assert len(PANDA_JOINT_NAMES) == len(self.panda_joint_idxs)
			self.cur_joints = np.zeros(len(PANDA_JOINT_NAMES), dtype=float)

		# Copy the joint values
		for i in range(len(self.panda_joint_idxs)):
			self.cur_joints[i] = msg.position[self.panda_joint_idxs[i]]

	def get_cur_fk(self):
		"""Get the current pose of the end effector"""
		if self.cur_joints is None:
			return None, None

		ee_pose = self.franka_kin.forward(list(self.cur_joints))
		ee_pose = np.asarray(ee_pose).reshape(3, 4)  # 3x4 rigid transformation matrix

		trans = ee_pose[:, -1]
		rot = R.from_matrix(ee_pose[:3, :3]).as_quat()

		return trans, rot

	def get_ik(self, trans, rot,  seed=None):
		"""Find joint values that bring the end effector
		to the specified pose

		:param trans: desired position of the end effector
		:param rot: desired orientation of the end effector
		:param seed: initial solution guess
		"""
		if self.cur_joints is None:
			return None

		mat_rot = R.from_quat(rot).as_matrix()
		ee_pose = np.concatenate([mat_rot, trans[:, np.newaxis]], axis=1)
		joint_configs = self.franka_kin.inverse(
			ee_pose.reshape(-1).tolist(),
			free_itr=10000,
			free_min=JOINT_MINS[-1],
			free_max=JOINT_MAXS[-1],
		)
		n_joints = len(PANDA_JOINT_NAMES)
		n_solutions = int(len(joint_configs) / n_joints)
		
		if n_solutions <= 0:
			return None

		joint_configs = np.asarray(joint_configs).reshape(n_solutions, n_joints)

		if seed is None:
			seed = self.cur_joints

		# Choose solution that is closest to current joint
		joint_idx = np.argmin(
		   
			np.sum(np.square(joint_configs - seed), axis=1)
		)

		return joint_configs[joint_idx]

	def _quat_shortest_path_angle(self, q1, q2):
		"""Find the minimum angle between two orientations
		Adapted from https://docs.ros.org/en/melodic/api/tf2/html/Quaternion_8h_source.html

		:param q1: the first orientation
		:param q2: the second orientation

		"""
		s = np.linalg.norm(q1) * np.linalg.norm(q2)

		# s = np.sqrt(np.linalg.norm(q1) * np.linalg.norm(q2))
		
		assert s > 0.0
		dot = np.sum(q1 * q2)
		if dot < 0:
			result = 2.0 * np.arccos(-dot / s)
		else:
			result = 2.0 * np.arccos(dot / s)

		assert result > 0.0
		return result


	def move_to_pose(self, trans_points, rot_points, traj_time = 5):
		"""Move the arm to the desired pose. Interpolate points in order to move along
		the shortest path from current pose to desired pose

		:param trans: Desired position of the end effector
		:param rot: Desired orientation of the end effector
		:param del_t: The amount of time to get to the desired pose
		:param trans_inc: The maximum distance between interpolated poses (in meters)
		:param rot_inc: The maximum angle between interpolated poses (in radians)

		"""
		
		joint_configs = np.empty((len(trans_points), 7), dtype=float)
		joint_seed = None

		for i in range(len(trans_points)):
			joint_configs[i] = self.get_ik(trans_points[i], rot_points[i], seed=joint_seed)
			joint_seed = joint_configs[i]


		# Define the total duration of the trajectory and the number of points
		traj_time = 20.0  # seconds
		num_points = len(trans_points)

		# Compute time intervals between points using a linspace
		time_intervals = np.linspace(0, traj_time, num_points)

		# Convert time intervals to cumulative times
		times = np.cumsum(time_intervals)

		return self.move(joint_configs, times)

	def move(self, path, times):
		"""Move the along the given joint trajectory

		:param path: Nx7 array of joint configurations to follow
		:param times: Array of size N at which joint configurations should be reached

		"""
		if isinstance(path, np.ndarray):
			path = path.tolist()

		assert len(path) > 0 and len(path) == len(times)

		goal_msg = FollowJointTrajectory.Goal()
		trajectory = JointTrajectory()
		trajectory.joint_names = PANDA_JOINT_NAMES

		trajectory.points = []
		for i in range(len(path)):
			assert len(path[i]) == 7
			point = JointTrajectoryPoint()
			point.positions = path[i]
			point.time_from_start.sec = int(times[i])
			point.time_from_start.nanosec = int(
				1e9 * (times[i] - point.time_from_start.sec)
			)
			trajectory.points.append(point)

		goal_msg.trajectory = trajectory
		self.action_client.wait_for_server()
		self.move_accepted = None
		self.move_result = None
		self.send_move_hook = self.action_client.send_goal_async(
			goal_msg, feedback_callback=self.move_feedback_cb
		)
		self.send_move_hook.add_done_callback(self.move_goal_cb)

		return

	def move_feedback_cb(self, feedback_msg):
		"""Get feedback from joint trajectory action server

		:param feedback_msg: Feedback info

		"""
		# feedback = feedback_msg.feedback
		# self.get_logger().info('Received feedback: {0}'.format(feedback))
		pass

	def move_goal_cb(self, future):
		goal_handle = future.result()
		self.move_accepted = goal_handle.accepted

		self.move_result_hook = goal_handle.get_result_async()
		self.move_result_hook.add_done_callback(self.move_result_cb)

	def move_result_cb(self, future):
		self.move_result = future.result().result

	def check_move_accepted(self):
		while self.move_accepted is None:
			pass
		return self.move_accepted

	def check_move_finished(self):
		return self.move_result is not None

	def check_move_success(self):
		return self.move_result is not None and self.move_result.error_code == 0

	def check_move_error_code(self):
		if self.move_result is None:
			return None
		else:
			return self.move_result.error_code


class MinimumJerkTrajectory:
	def __init__(self, start, end, duration, dt):
		self.start = start
		self.end = end
		self.duration = duration
		self.dt = dt

	def generate_trajectory(self):
		t = np.arange(0, self.duration, self.dt)
		n = len(t)

		# Generate minimum jerk trajectory coefficients for each dimension
		coeffs = []
		for i in range(len(self.start)):
			a0 = self.start[i]
			a1 = 0
			a2 = 0
			a3 = (20 * (self.end[i] - self.start[i])) / (2 * self.duration**3)
			a4 = -(30 * (self.end[i] - self.start[i])) / (2 * self.duration**4)
			a5 = (12 * (self.end[i] - self.start[i])) / (2 * self.duration**5)
			coeffs.append([a0, a1, a2, a3, a4, a5])

		# Calculate trajectory points for each dimension using the coefficients
		trajectory = np.zeros((n, len(self.start)))
		for i in range(len(self.start)):
			trajectory[:, i] = np.polyval(coeffs[i], t)

		return trajectory


class RotationSlerp:
	def __init__(self, start_rotation, end_rotation, duration, dt):
		self.start_rotation = start_rotation
		self.end_rotation = end_rotation
		self.duration = duration
		self.dt = dt

	def slerp(self, q1, q2, t):
		# Spherical Linear Interpolation (slerp) between two quaternions
		dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
		omega = np.arccos(dot_product)
		sin_omega = np.sin(omega)

		if np.isclose(sin_omega, 0):
			return q1

		q_interp = (np.sin((1 - t) * omega) / sin_omega) * q1 + (np.sin(t * omega) / sin_omega) * q2

		return q_interp

	def generate_trajectory(self):
		t = np.arange(0, self.duration, self.dt)

		# Interpolate using slerp
		rotation_trajectory = np.array([self.slerp(self.start_rotation, self.end_rotation, ti) for ti in t])

		return rotation_trajectory
	

class CombinedTrajectory:
	def __init__(self, translation_traj, rotation_traj):
		self.translation_traj = translation_traj
		self.rotation_traj = rotation_traj

	def get_combined_trajectory(self):
		combined_trajectory = np.hstack((self.translation_traj, self.rotation_traj))
		return combined_trajectory


def get_des_pose():

	stop_sig = False

	key = input("Enter cmd")
	if key == "a":
		out_trans = [0,0,0.5]
		out_rot  =  []
		
	elif key == "n":
		stop_sig = True
	else:
		print("Key not recognized")

	# out_rot = R.from_euler("xyz", out_rot, degrees=False).as_quat()

	return out_trans, out_rot, stop_sig


def main(args=None):
	
	JOINT_CMD_TOPIC = "joint_trajectory_controller/follow_joint_trajectory"
	JOINT_STATES_TOPIC = "joint_states"

	rclpy.init(args=args)
	franka_robot = FrankaRobot(
		joint_cmd_topic=JOINT_CMD_TOPIC, joint_states_topic=JOINT_STATES_TOPIC
	)

	thread = Thread(target=rclpy.spin, args=(franka_robot,))
	thread.start()

	# Wait until we are ready to do fk (need joint state message)
	while franka_robot.get_cur_fk()[0] is None:
		pass

	if franka_robot.get_cur_fk()[0] is not None:

		start_translation, start_rotation = franka_robot.get_cur_fk()

		if start_translation is None or start_rotation is None:
			print("Failed to get fk")
			
		print("Cur trans: {}, rot {}".format(start_translation, start_rotation))
		print("Cur joints: {}".format(franka_robot.cur_joints))

		end_translation, end_rotation, stop_signal = get_des_pose()
		duration_translation = 20.0
		dt_translation = 0.1
		translation_trajectory = MinimumJerkTrajectory(start_translation, end_translation, duration_translation, dt_translation).generate_trajectory()

		# start_rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
		# end_rotation = Rotation.from_euler('xyz', [45, 90, 180], degrees=True)
		print(translation_trajectory)

		duration_rotation = 20.0
		dt_rotation = 0.1

		end_rotation = start_rotation

		rotation_trajectory = RotationSlerp(start_rotation, end_rotation, duration_rotation, dt_rotation).generate_trajectory()

		# combined_trajectory = CombinedTrajectory(translation_trajectory, rotation_trajectory)
		# combined_traj_data = combined_trajectory.get_combined_trajectory()

		franka_robot.move_to_pose(translation_trajectory, rotation_trajectory)

		# Wait until robot has finished moving
		while not franka_robot.check_move_finished():
			pass

	franka_robot.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
	main()
