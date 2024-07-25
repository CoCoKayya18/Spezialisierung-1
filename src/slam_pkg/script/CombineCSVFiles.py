import pandas as pd
import os 


def combine_csv_files(paths, filenames, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in filenames:
        combined_df = pd.DataFrame()
        
        for path in paths:
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            else:
                print(f"File {file_path} does not exist.")
        
        # Save the combined DataFrame to the output directory
        output_file_path = os.path.join(output_dir, filename)
        combined_df.to_csv(output_file_path, index=False)
        print(f"Combined file saved to {output_file_path}")

filenames = ['FullData.csv', 'GT_Deltas.csv', 'Vels_And_Accels.csv']

combPathsSquare = [
    '../Spezialisierung-1/src/slam_pkg/data/x_direction_positive',
    '../Spezialisierung-1/src/slam_pkg/data/x_direction_negative',
    '../Spezialisierung-1/src/slam_pkg/data/y_direction_positive',
    '../Spezialisierung-1/src/slam_pkg/data/y_direction_negative'
]

combPathsDiagonal_first_and_third_quad = [
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_first_quad',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_third_quad'
]

combPathsDiagonal_second_and_fourth_quad = [
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_second_quad',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_fourth_quad'
]

combPathsAll = [
    '../Spezialisierung-1/src/slam_pkg/data/x_direction_positive',
    '../Spezialisierung-1/src/slam_pkg/data/x_direction_negative',
    '../Spezialisierung-1/src/slam_pkg/data/y_direction_positive',
    '../Spezialisierung-1/src/slam_pkg/data/y_direction_negative',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_first_quad',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_second_quad',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_third_quad',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_fourth_quad'
]

output_dirSquare = '../Spezialisierung-1/src/slam_pkg/data/square'
output_dirFirstAndThird = '../Spezialisierung-1/src/slam_pkg/data/diagonal_first_and_third_quad'
output_dirSecondAndFourth = '../Spezialisierung-1/src/slam_pkg/data/diagonal_second_and_fourth_quad'
output_dirAll = '../Spezialisierung-1/src/slam_pkg/data/AllCombined'

combine_csv_files(combPathsSquare, filenames, output_dirSquare)
combine_csv_files(combPathsDiagonal_first_and_third_quad, filenames, output_dirFirstAndThird)
combine_csv_files(combPathsDiagonal_second_and_fourth_quad, filenames, output_dirSecondAndFourth)
combine_csv_files(combPathsAll, filenames, output_dirAll)
