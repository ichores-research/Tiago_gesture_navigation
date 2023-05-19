H36M_JOINTS_DESC = ['root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

class LocationPoint:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

class Location3DJoint:
    def __init__(self, x=None, y=None, z=None):
        self.x=x
        self.y=y
        self.z=z


class HumanInfo:
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
        for joint in self.motionbert_joints_positions:
            coordinates = self.motionbert_joints_positions[joint]
            print("%s:(%f, %f, %f)" %(joint, coordinates.x, coordinates.y, coordinates.z))

    def create_joints_dict(self):
        joint_idx = 0
        joints_dict = {}
        for joint in H36M_JOINTS_DESC:
            joints_dict[joint] = list(self.motionbert_joints_positions[0][joint_idx])
            joint_idx += 1
        self.motionbert_joints_positions = joints_dict
        for joint in self.motionbert_joints_positions:
            coordinates = self.motionbert_joints_positions[joint]
            self.motionbert_joints_positions[joint] = Location3DJoint(
                x=coordinates[0],
                y=coordinates[2],
                z=coordinates[1])

    def change_reference_of_joints(self):
        """
        Funkce posouvajici pocatek souradneho systemu bodu.
        """
        center = Location3DJoint()
        left_hip = self.motionbert_joints_positions["LHip"]
        print(left_hip.x)
        right_hip = self.motionbert_joints_positions["RHip"]
        print(right_hip.x)
        left_ankle = self.motionbert_joints_positions["LAnkle"]
        right_ankle = self.motionbert_joints_positions["RAnkle"]
        center.x = (left_hip.x + right_hip.x)/2
        center.y = (left_hip.y + right_hip.y)/2
        center.z = (left_ankle.z + right_ankle.z)/2
        print("Center location: (%f, %f, %f)" %(center.x, center.y, center.z))
        for joint in self.motionbert_joints_positions:
            self.motionbert_joints_positions[joint].x -= center.x
            self.motionbert_joints_positions[joint].y -= center.y
            self.motionbert_joints_positions[joint].z -= center.z
            self.motionbert_joints_positions[joint].z = -self.motionbert_joints_positions[joint].z # chci aby nahoru bylo do kladnych hodnot -> pravotocivy souradny system.
        self.clean_print_joints()