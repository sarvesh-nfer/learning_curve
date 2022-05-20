from pickletools import anyobject
import socket
import time
import pandas as pd
import argparse
import os

# Arguments for the script running

parser = argparse.ArgumentParser()
parser.add_argument('--station', metavar='station', type= int, help='Enter Station Number')
args = parser.parse_args()


# Binding of the server 
host = "10.20.0.1"
port = 30002


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((host, port))
server.listen(5)
client , addr = server.accept()
print('Connected with address:', addr)

# reading path data
directory = 'data\\eng_csv'
station = args.station
mov1 = pd.read_csv(os.path.join(directory, 'mov1_s{}.csv'.format(station)))
mov2 = pd.read_csv(os.path.join(directory, 'mov2_s{}.csv'.format(station)))
drop1 = pd.read_csv(os.path.join(directory, 'drop1_s{}.csv'.format(station)))
drop2 = pd.read_csv(os.path.join(directory, 'drop2_s{}.csv'.format(station)))
reject1 = pd.read_csv(os.path.join(directory, 'reject1_s{}.csv'.format(station)))
reject2 = pd.read_csv(os.path.join(directory, 'reject2_s{}.csv'.format(station)))

# print('mov1:')
# print(mov1)
# print('mov2:')
# print(mov2)
# print('drop1:')
# print(drop1)
# print('drop2:')
# print(drop2)
# print('reject1:')
# print(reject1)
# print('reject2:')
# print(reject2)

motion_paths = [
    mov1,
    mov2,
    mov1,
    drop1,
    drop2,
    mov1,
    reject1,
    reject2,
]

aj = 10
vj = 20
al = 1.2
vl = 0.6

cycle = 0
while cycle < 10:
    cycle = cycle + 1

    try:
        # Executing the paths in list motion_paths
        for i, paths in enumerate(motion_paths):
            print(motion_paths[i])
            trajectory = []
            # Converting data into list format
            for index in range(len(paths)):
                waypoint = paths.iloc[index]
                if waypoint['MOVELJ'] == 0:
                    a = aj
                    v = vj
                if waypoint['MOVELJ'] == 1:
                    a = al
                    v = vl
                points = []
                points.append(waypoint['PX'])
                points.append(waypoint['PY'])
                points.append(waypoint['PZ'])
                points.append(waypoint['RX'])
                points.append(waypoint['RY'])
                points.append(waypoint['RZ'])
                points.append(a)
                points.append(v)
                points.append(0)
                points.append(0)
                points.append(int(waypoint['MOVELJ']))
                # print(points)
                trajectory.append(points)
            # print(trajectory)

            if i == 3:
                points = []
                waypoint = paths.iloc[index]
                points.append(waypoint['PX'])
                points.append(waypoint['PY'])
                points.append(waypoint['PZ'] - 0.1)
                points.append(waypoint['RX'])
                points.append(waypoint['RY'])
                points.append(waypoint['RZ'])
                points.append(a)
                points.append(v)
                points.append(0)
                points.append(0)
                points.append(int(waypoint['MOVELJ']))
                # print(points)
                trajectory.append(points)
            # print(trajectory)

            msg = client.recv(1024)
            print(msg)
            msg = msg.decode('utf8') 
            print(msg)

            # time.sleep(1)


            if msg == "asking_for_trajectory":
                print("sending trajectory point count")
                client.send(("({})".format(len(trajectory))).encode('utf8'))

                
            motion = True
            for point in trajectory:
                print('Setting acceleration: {} Velocity: {}'.format(point[6], point[7]))
                sd_point = "({},{},{},{},{},{},{},{},{},{},{})".format(
                    point[0],
                    point[1],
                    point[2],
                    point[3],
                    point[4],
                    point[5],
                    point[6],
                    point[7],
                    point[8],
                    point[9],
                    point[10],
                )
                if motion:
                    motion = False
                    start = time.time()
                    print('Sending waypoint:',point)
                    client.send((sd_point).encode('utf8'))
                    msg = client.recv(1024)
                    msg = msg.decode('utf8')
                    print('recieved message:', msg)

                    if msg == 'C':
                        stop = time.time()
                        motion = True
                        msg = None
                print('Time taken for the path: {}sec'.format(stop - start))

            time.sleep(1)
                
    except Exception as error:
        print(error)
        # break

client.close()
server.close()

