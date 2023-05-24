import math

H36M_JOINTS_DESC = ['root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

DATA_SCALE = 0.9 # MotionBERT returns joint position on z-axis between [0,2] -> scaling down to human with height 1.8m

class LocationPoint:
    """
    Object serving as a simple 2D point
    """
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

class Location3DJoint:
    """
    Object serving as a simple 3D point
    """
    def __init__(self, x=None, y=None, z=None):
        self.x=x
        self.y=y
        self.z=z

class HumanInfo:
    """
    Class containing information about human located in the picture
    """
    def __init__(self):
        self.human_found = False
        self.human_centered = False
        self.center_location = LocationPoint()
        self.final_robot_rotation = LocationPoint()
        self.distance_from_robot = None
        self.alphapose_json_result = None
        self.motionbert_joints_positions = None

    def is_found(self):
        return self.human_found
    
    def is_centered(self):
        return self.human_centered
    
    def clean_print_joints(self):
        """
        Function for printing found joints nicely 
        """
        for joint in self.motionbert_joints_positions:
            coordinates = self.motionbert_joints_positions[joint]
            print("%s:  (%f, %f, %f)" %(joint, coordinates.x, coordinates.y, coordinates.z))

    def create_joints_dict(self):
        """
        Function that creates dictionary of joints from data provided by MotionBERT 
        and rescales them to match size of an average human.
        """
        joint_idx = 0
        joints_dict = {}
        for joint in H36M_JOINTS_DESC:
            joints_dict[joint] = list(self.motionbert_joints_positions[0][joint_idx])
            joint_idx += 1
        self.motionbert_joints_positions = joints_dict
        for joint in self.motionbert_joints_positions:
            coordinates = self.motionbert_joints_positions[joint]
            self.motionbert_joints_positions[joint] = Location3DJoint(
                x=coordinates[0] * DATA_SCALE,
                y=coordinates[2] * DATA_SCALE,
                z=coordinates[1] * DATA_SCALE
            )

    def change_reference_of_joints(self):
        """
        Function that moves origin of coordinates of human joints to place on the ground between the hips
        """
        center = Location3DJoint()

        left_hip = self.motionbert_joints_positions["LHip"]
        right_hip = self.motionbert_joints_positions["RHip"]
        left_ankle = self.motionbert_joints_positions["LAnkle"]
        right_ankle = self.motionbert_joints_positions["RAnkle"]
        
        center.x = (left_hip.x + right_hip.x)/2
        center.y = (left_hip.y + right_hip.y)/2
        center.z = min(left_ankle.z, right_ankle.z)
        
        for joint in self.motionbert_joints_positions:
            self.motionbert_joints_positions[joint].x -= center.x
            self.motionbert_joints_positions[joint].y -= center.y
            self.motionbert_joints_positions[joint].z -= center.z
            # We want to have up direction from ground to be in positive coordinates -> righthand coordinate system.
            self.motionbert_joints_positions[joint].z = -self.motionbert_joints_positions[joint].z

        right_wrist = self.motionbert_joints_positions["RWrist"]
        right_wrist.y = -abs(right_wrist.y) # hand is always in front of the human, not behind him

        if abs(right_wrist.x) < 0.3:
            right_wrist.y -= math.sqrt(math.pow(0.3, 2) - math.pow(right_wrist.x, 2))

        self.clean_print_joints()