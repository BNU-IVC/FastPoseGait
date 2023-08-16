import numpy as np
import logging


class GraphPartition():
    """
    Support for the joint format of coco17.
    """
    def __init__(self, joint_format='coco'):

        if joint_format == 'coco':
            whole_body = [list(range(0,17))]
            self.partition_list = []
            parts5 = [
                np.array([5, 7, 9]),         # left_arm
                np.array([6, 8, 10]),        # right_arm
                np.array([11, 13, 15]),      # left_leg
                np.array([12, 14, 16]),      # right_leg
                np.array([0, 1, 2, 3, 4]),   # head
            ]
            parts2_1 = [
                np.array([5, 7, 9, 6, 8, 10, 0, 1, 2, 3, 4]), # top_half
                np.array([11, 13, 15, 12, 14, 16])            # bottom_half
            ]
            parts2_2 = [
                np.array([5, 7, 9, 6, 8, 10]),                    # arms
                np.array([11, 13, 15, 12, 14, 16, 0, 1, 2, 3, 4]) # head + legs
            ]
            parts3_1 = [
                np.array([5, 7, 9, 6, 8, 10]),      # arms
                np.array([11, 13, 15, 12, 14, 16]), # legs
                np.array([0, 1, 2, 3, 4])           # head
            ]
            parts3_2 = [
                np.array([5, 7, 9, 12, 14, 16]),  # left_arm + right_leg
                np.array([6, 8, 10, 11, 13, 15]), # right_arm + left_leg
                np.array([0, 1, 2, 3, 4])         # head
            ]
            self.partition_list.append(whole_body)
            self.partition_list.append(parts5)
            self.partition_list.append(parts2_1)
            self.partition_list.append(parts2_2)
            self.partition_list.append(parts3_1)
            self.partition_list.append(parts3_2)

        elif joint_format == 'coco-no-head':
            self.partition_list = []
            whole_body = [list(range(0,12))]
            parts4 = [
                np.array([0, 2, 4]),     # left_arm
                np.array([1, 3, 5]),     # right_arm
                np.array([6, 8, 10]),    # left_leg
                np.array([7, 9, 11]),    # right_leg
            ]
            parts2_1 = [
                np.array([0, 2, 4, 1, 3, 5]),  # top_half
                np.array([6, 8, 10, 7, 9, 11]) # bottom_half
            ]
            parts2_2 = [
                np.array([0, 2, 4, 7, 9, 11]),  # left_arm + right_leg
                np.array([1, 3, 5, 6, 8, 10]),  # right_arm + left_leg
            ]
            self.partition_list.append(whole_body)
            self.partition_list.append(parts4)
            self.partition_list.append(parts2_1)
            self.partition_list.append(parts2_2)

    def __call__(self, part=None):
        if part:
            if not isinstance(part, int):
                logging.error('Error: give a integer for partition!')
                raise ValueError()
            ret_partition = self.partition_list[part]
        else:
            ret_partition = self.partition_list[0]
        return ret_partition