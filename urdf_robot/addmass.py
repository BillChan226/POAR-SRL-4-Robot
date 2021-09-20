import re


def add_mass():
    pattern = re.compile('(<link name=\".*>)')
    mass = '''    <inertial>
          <origin rpy="0 0 0" xyz="0 0 0.055"/>
          <!--Increase mass from 5 Kg original to provide a stable base to carry the
              arm.-->
          <mass value="1.00"/>
          <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.06" iyz="0" izz="0.03"/>
        </inertial>'''
    with open("inmoov_col.urdf","r") as f:
        string = f.read()
        string = re.sub(pattern,r'\1\n'+mass, string)
        # print(string)
    with open("inmoov_colmass.urdf","w") as f:
        f.write(string)


add_mass()
