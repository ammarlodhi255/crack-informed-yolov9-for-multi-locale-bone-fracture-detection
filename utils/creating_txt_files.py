import os

def copy_filenames_to_txt(source_folder, output_file):
    # List all files in the source folder
    files = os.listdir(source_folder)
    
    # Create or open the output text file in write mode
    with open(output_file, "w") as txt_file:
        # Write each file name to the text file, one per line
        for file in files:
            txt_file.write(os.path.join(source_folder, file) + "\n")

# Example usage:
source_folder = "/cluster/home/ammaa/Downloads/Ammars/Models/Fracture_Detection/YOLOV9/data/Doubled/images/test/"
output_file = "/cluster/home/ammaa/Downloads/Ammars/Models/Fracture_Detection/YOLOV9/data/Doubled/test.txt"

copy_filenames_to_txt(source_folder, output_file)
