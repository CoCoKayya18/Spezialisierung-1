import pandas as pd
import os 

def combine_csv_files(paths, filenames, output_dir, single=False):
    suffix = "_single" if single else ""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in filenames:
        combined_df = pd.DataFrame()
        
        for path in paths:
            file_path = os.path.join(path, filename.replace('.csv', f'{suffix}.csv'))
            print(f"Checking for file: {file_path}")  # Debug statement
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if df.empty:
                    print(f"Warning: {file_path} is empty.")  # Debug statement
                else:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                    print(f"Added {len(df)} rows from {file_path}")  # Debug statement
            else:
                print(f"File {file_path} does not exist.")
        
        # Save the combined DataFrame to the output directory
        output_file_path = os.path.join(output_dir, filename.replace('.csv', f'{suffix}.csv'))
        combined_df.to_csv(output_file_path, index=False)
        print(f"Combined file saved to {output_file_path} with {len(combined_df)} rows")  # Debug statement

filenames = ['FullData.csv', 'GT_Deltas.csv', 'Vels_And_Accels.csv']

combPathsSquare = [
    '../Spezialisierung-1/src/slam_pkg/data/x_direction_positive',
    '../Spezialisierung-1/src/slam_pkg/data/x_direction_negative',
    '../Spezialisierung-1/src/slam_pkg/data/y_direction_positive',
    '../Spezialisierung-1/src/slam_pkg/data/y_direction_negative'
]

combPathsSquareSingle = [
    '../Spezialisierung-1/src/slam_pkg/data/x_direction_positive_single',
    '../Spezialisierung-1/src/slam_pkg/data/x_direction_negative_single',
    '../Spezialisierung-1/src/slam_pkg/data/y_direction_positive_single',
    '../Spezialisierung-1/src/slam_pkg/data/y_direction_negative_single'
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

combPathsDiagonal_first_and_third_quad_single = [
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_first_quad_single',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_third_quad_single'
]

combPathsDiagonal_second_and_fourth_quad_single = [
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_second_quad_single',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_fourth_quad_single'
]

combPathsAllSingle = [
    '../Spezialisierung-1/src/slam_pkg/data/x_direction_positive_single',
    '../Spezialisierung-1/src/slam_pkg/data/x_direction_negative_single',
    '../Spezialisierung-1/src/slam_pkg/data/y_direction_positive_single',
    '../Spezialisierung-1/src/slam_pkg/data/y_direction_negative_single',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_first_quad_single',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_second_quad_single',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_third_quad_single',
    '../Spezialisierung-1/src/slam_pkg/data/diagonal_fourth_quad_single'
]

output_dirSquare = '../Spezialisierung-1/src/slam_pkg/data/square'
output_dirSquare_single = '../Spezialisierung-1/src/slam_pkg/data/square_single'
output_dirFirstAndThird = '../Spezialisierung-1/src/slam_pkg/data/diagonal_first_and_third_quad'
output_dirSecondAndFourth = '../Spezialisierung-1/src/slam_pkg/data/diagonal_second_and_fourth_quad'
output_dirAll = '../Spezialisierung-1/src/slam_pkg/data/AllCombined'
output_dirFirstAndThird_single = '../Spezialisierung-1/src/slam_pkg/data/diagonal_first_and_third_quad_single'
output_dirSecondAndFourth_single = '../Spezialisierung-1/src/slam_pkg/data/diagonal_second_and_fourth_quad_single'
output_dirAllSingle = '../Spezialisierung-1/src/slam_pkg/data/AllCombined_single'

combine_csv_files(combPathsSquare, filenames, output_dirSquare)
combine_csv_files(combPathsSquareSingle, filenames, output_dirSquare_single, single=True)
combine_csv_files(combPathsDiagonal_first_and_third_quad, filenames, output_dirFirstAndThird)
combine_csv_files(combPathsDiagonal_second_and_fourth_quad, filenames, output_dirSecondAndFourth)
combine_csv_files(combPathsAll, filenames, output_dirAll)
combine_csv_files(combPathsAllSingle, filenames, output_dirAllSingle, single=True)
combine_csv_files(combPathsDiagonal_first_and_third_quad_single, filenames, output_dirFirstAndThird_single, single=True)
combine_csv_files(combPathsDiagonal_second_and_fourth_quad_single, filenames, output_dirSecondAndFourth_single, single=True)
