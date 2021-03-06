'''
This is where data will be sent to the arduino to control our robot.
'''

import serial
import time


def est_serial_cnxn():
    try:
        ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1)
        if ser.is_open:
            print("Connection ACM0 opened..")
    except:
        ser = serial.Serial("/dev/ttyACM1", 9600, timeout=1)
        if ser.is_open:
            print("Connection ACM1 opened..")
    return ser


def send(ser, data):
    #sent = False
    #while not sent:
    if 0 <= int(data) <= 300 or int(data) == 999:
        #line = str(ser.readline())
        #print("line in send is: {}".format(line)) this was reading partial vals
        sendBuff = ("<{}<".format(data))
        print("Data sent: {}".format(sendBuff))
        ser.write(sendBuff.encode(encoding='ascii'))
        #print("sent: {}".format(data))
        # sent = True
        '''
        if "moving" not in line:
            ser.write(data.encode(encoding='ascii'))
            print("Sent: " + data)
            sent = True
        elif "moving" in line:
            print("(E) Hockey stick is currently moving... data not sent.")
        '''
    #else:
    #print("(E) Data: {} out of Range.. not sent.".format(data))
    


    
