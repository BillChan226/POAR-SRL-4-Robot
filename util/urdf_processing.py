import re
import os
from ipdb import set_trace as tt
import re

urdf_path = "/home/tete/work/SJTU/RL-InmoovRobot/urdf_robot/jaka_urdf/"
file_path = os.path.join(urdf_path, "jaka.urdf")
new_file = os.path.join(urdf_path, "jaka_local.urdf")

def read_file(path):
    """
    小心，這個東西會吃內存，太大的文件不要搞
    :param path:
    :return:
    """
    file_object = open(path)
    file_content = file_object.read()
    file_split = file_content.splitlines()
    return file_split

def block_finder_by_marker(content, marker):
    """abandoned, not useful, since the 'link' can be only a half complete"""
    assert isinstance(content, list), "Please feed me with list"
    stack = 0
    block_index = [''] * len(content)
    num_block = 0
    filenames = []
    stacks = [0] * len(content)
    for i, s in enumerate(content):
        stacks[i] = stack
        if stack == 1:
            block_index[i] = "block"
            if "filename" in s:
                filenames.append(s.split("\"")[1])  # mesh name
        if "<"+marker in s:
            stack += 1
            block_index[i] = "start"
        elif "</"+marker in s:
            stack -= 1
            block_index[i] = "end"
            num_block +=1
    print("Number of blocks found: {}".format(num_block))
    return block_index, filenames, stacks


def block_finder_by_filename(content, marker):
    assert isinstance(content, list), "Please feed me with list"
    filename = []
    block_index = [''] * len(content)
    origin = []
    for i, s in enumerate(content):
        j, k = i, i
        if "filename" in s:
            filename.append(s[s.index('<'):])
            # while marker not in content[j] and j > 1:
            #     block_index[j] = 'b'
            #     j -= 1

            # 向下找marker，在link的地方標記 b
            while marker not in content[k] and k <= len(content) - 2:
                k += 1

            #To find the origin
            m = i
            while 'visual' not in content[m]:
                m -=1
                if 'origin' in content[m]:
                    origin.append(content[m])
            m = i
            while 'visual' not in content[m]:
                m +=1
                if 'origin' in content[m]:
                    origin.append(content[m])
            block_index[k-1] = 'b'
    return block_index, filename, origin

def add_collision(content, block_index, filename, origin):
    col_list = ['skull']
    new_content = []
    num_block = 0
    for i in range(len(content)):
        if block_index[i] != 'b':
            new_content.append(content[i])
        else:
            s = content[i]
            new_content.append(s)
            # flag = False
            # for ele in col_list:
            #     if ele in filename[num_block]:
            #         flag = True
            # if flag:
            space = s.split('<')[0]
            new_content.append(space + '<collision>')
            new_content.append(space + '  {}'.format(origin[num_block][6:]))
            new_content.append(space + '  <geometry>')
            new_content.append(space + '    {}'.format(filename[num_block]))
            new_content.append(space + '  </geometry>')
            new_content.append(space + '</collision>')
            # new_content.append(space + '<inertial>')
            # new_content.append(space + '  <origin rpy="0 0 0" xyz="0 0 0"/>')
            # new_content.append(space + '  <mass value="0.0"/>')
            # new_content.append(space + '  <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.06" iyz="0" izz="0.03"/>')
            # new_content.append(space + '</inertial>')
            num_block += 1
    return new_content

def replace_file_name_to_relative(data):
    local_data = []
    for line in data:
        if "filename" in line:
            new_line = line.split("package://jaka_ur_description_pkg/")
            print("".join(new_line))
            local_data.append("".join(new_line))
        else:
            local_data.append(line)
    return local_data

def save_list_to_file(file_name, data):
    if os.path.exists(file_name):
        os.remove(file_name)
    message = ''
    for s in data:
        message += s + '\n'
    file = open(file_name, 'w')
    file.write(message)
    file.close()

if __name__ == "__main__":
    message = read_file(file_path)
    new_message_jaka = replace_file_name_to_relative(message)
    # blocks, mesh_file, cord_origin = block_finder_by_filename(message, "link")
    # new_message = add_collision(message, blocks, mesh_file, cord_origin)
    save_list_to_file(new_file, new_message_jaka)

