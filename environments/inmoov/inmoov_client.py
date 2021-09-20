import zmq
from zmq import ssh
import numpy as np

from environments.inmoov.inmoov_p2p_client_ready import InmoovGymEnv
from .inmoov_server import server_connection, client_ssh_connection, client_connection

SERVER_PORT = 7777
HOSTNAME = 'localhost'

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def test_inmoov_gym():
    while True:
        k = input()
        try:
            # time.sleep(0.5)
            action = np.zeros(shape=(joints_num,))
            signal = k.split()
            joint, move = int(signal[0]), float(signal[1])
            action[joint] = move
            robot.step(action)
        except:
            continue
        # robot.step()

if __name__ == "__main__":
    socket = server_connection()
    robot = InmoovGymEnv(debug_mode=True, positional_control=True)
    init_pose = robot._inmoov.get_joints_pos()
    joints_num = len(init_pose)

    while True:
        msg = socket.recv_json()
        command = msg["command"]
        if command == "position":
            data = robot.server_step(msg[command])
            joint_state, reward, done, infos, px, end_position = data
            send_array(socket, joint_state, flags=0, copy=True, track=False)
            send_array(socket, np.array(reward), flags=0, copy=True, track=False)
            send_array(socket, np.array(done), flags=0, copy=True, track=False)
            send_array(socket, px, flags=0, copy=True, track=False)
            send_array(socket, end_position, flags=0, copy=True, track=False)
            print("message sent")
        elif command == "action":
            print(1)
        elif command == "done":
            print(2)
        elif command == "reset":
            print(3)
