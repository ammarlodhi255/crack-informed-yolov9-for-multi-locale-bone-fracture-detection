input_file = "/cluster/home/ammaa/Downloads/Ammars/Models/Fracture_Detection/yolov9-main/data/Split-Authors/test.txt"
output_file = "/cluster/home/ammaa/Downloads/Ammars/Models/Fracture_Detection/yolov9-main/data/Split-Authors/test-new.txt"

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        filename = line.strip()
        new_filename = "/cluster/home/ammaa/Downloads/Ammars/Models/Fracture_Detection/yolov9-main/data/Split-Authors/images/test/" + filename
        f_out.write(new_filename + '\n')
