# 输出需要训练或测试的文件到./lists/lists_liver/train.txt和test_vol.txt, 一般只生成一次，其他时候用来debug
import os

def write_filenames_to_txt(input_dir, output_dir,output_file, end_with='.npz', max_files=None):
    # 获取所有npz文件
    file_names = [f.split('.')[0] for f in os.listdir(input_dir) if f.endswith(end_with)]

    # 如果指定了最大文件数，限制文件列表的长度
    if max_files is not None:
        file_names = file_names[:max_files]

    # 写入到文件
    with open(os.path.join(output_dir, output_file), 'w') as file:
        for name in file_names:
            file.write(name + '\n')

# 输出train文件名
input_directory = ''
output_dir = ''
output_file_name = 'train.txt'

write_filenames_to_txt(input_directory, output_dir,output_file_name,max_files=max_files)
### 输出测试集文件名
input_directory = ''
output_dir = ''
output_file_name = 'test_vol.txt'
write_filenames_to_txt(input_directory, output_dir,output_file_name, end_with='.h5',max_files=max_files)