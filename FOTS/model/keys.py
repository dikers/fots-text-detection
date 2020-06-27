import os
import glob


def get_key_from_file_list(input_dir):
    file_list = glob.glob(os.path.join(input_dir, '*.txt'))

    #print(file_list)

    content = ''
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                content += line

    content = content.replace('\n', '')
    content = content.replace('-', '一')
    char_list = list(set(content))
    char_list.sort()
    content = ''.join(char_list)
    print("-"*30)
    print('class nums: ',  len(content)+1)
    print(content)
    print("-"*30)
    return content

