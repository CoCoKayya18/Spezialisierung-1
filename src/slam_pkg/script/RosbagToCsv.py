import pandas as pd
from bagpy import bagreader
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

class BagDataProcessor:
    def __init__(self, bag_file_path):
        self.bag = bagreader(bag_file_path)

    def read_topic_to_dataframe(self, topic_name):
        csv_path = self.bag.message_by_topic(topic_name)
        return pd.read_csv(csv_path) if csv_path else pd.DataFrame()

    def process_ground_truth_data(self, df):
        # Here you could calculate deltas, velocities, or accelerations for ground truth data.
        # Placeholder for your processing logic.
        return df

    def process_joint_state_data(self, df):
        # Similar to the ground truth processing, this is where you could calculate deltas for joint states.
        # Placeholder for your processing logic.
        return df

    def merge_and_process_data(self, ground_truth_df, joint_state_df):
        # Assuming both dataframes have a 'Time' column for merging.
        merged_df = pd.merge_asof(ground_truth_df.sort_values('Time'), joint_state_df.sort_values('Time'), on='Time')

        # After merging, you can process the merged data as needed.
        return merged_df

    def save_to_csv(self, df, file_name):
        df.to_csv(file_name, index=False)

def process_bag_file(bag_file_path):

    processor = BagDataProcessor(bag_file_path)

    # Read data from specified topics.
    ground_truth_df = processor.read_topic_to_dataframe('ground_truth/state')
    joint_state_df = processor.read_topic_to_dataframe('joint_state')

    # Process each topic's data as needed.
    ground_truth_df = processor.process_ground_truth_data(ground_truth_df)
    joint_state_df = processor.process_joint_state_data(joint_state_df)

    # Merge and further process the data from different topics.
    merged_df = processor.merge_and_process_data(ground_truth_df, joint_state_df)

    # Define the output file path.
    output_file_path = os.path.splitext(bag_file_path)[0] + '_processed.csv'
    processor.save_to_csv(merged_df, output_file_path)

    print(f"Processed data saved to: {output_file_path}")

if __name__ == '__main__':
    
    # Example usage: process each bag file provided as input separately.
    bag_files = ['/path/to/your/first.bag', '/path/to/your/second.bag']

    for bag_file in bag_files:
        process_bag_file(bag_file)
