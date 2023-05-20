import numpy as np
import pandas as pd
import math

#
Data ='''10 23  3  1  8  0  0.0-0.267112627625D-04-0.159161572810D-11 0.000000000000D+00
    0.850000000000D+02-0.957187500000D+02 0.408445584827D-08-0.223802637826D+01
   -0.503845512867D-05 0.845144712366D-02 0.699050724506D-05 0.515369205856D+04
    0.288000000000D+06 0.113621354103D-06-0.135885460275D+01-0.763684511185D-07
    0.977219690448D+00 0.250718750000D+03-0.243635316553D+01-0.779425323336D-08
    0.121433629627D-09 0.100000000000D+01 0.225100000000D+04 0.000000000000D+00
    0.200000000000D+01 0.000000000000D+00 0.232830643654D-08 0.850000000000D+02
    0.280874000000D+06 0.400000000000D+01 0.000000000000D+00 0.000000000000D+00'''

# Constants
c= 299792458 #𝑚/𝑠
u= 3.986005*(10**14) #𝑚3/𝑠2
wE= 7.2921151467*(10**-5) #𝑟𝑎𝑑/𝑠
a = 6378137 # m

epoch = 17 * 1960 
Prn_number = 10

def cal_brd(eph = epoch , brd = Data):
    def determine_rotation_matrices(angle , axis):
        # this means we are working with the x axis
        if axis == 1:
            rotation_matrices = np.array([[1 , 0 , 0], [0 , math.cos(angle) ,math.sin(angle)],[0 , -math.sin(angle),math.cos(angle)]])

            return rotation_matrices

        # this means we are working with the y axis
        elif axis == 2:
            rotation_matrices = np.array([[math.cos(angle),0,-math.sin(angle)], [0,1,0], [math.sin(angle),0, math.cos(angle)]])

            return rotation_matrices

        # this means we are working with the z axis
        elif axis == 3:
            rotation_matrices = np.array([[math.cos(angle),math.sin(angle),0], [-math.sin(angle),math.cos(angle),0], [0,0,1]])

            return rotation_matrices

    # this is the main function we are umath.sing np.dot to do the matrix multipication 
    def rotation(vector,angle,axis):
        if vector.shape == (3,) :
            vector = np.transpose(vector)
            rotation_matrix = determine_rotation_matrices(angle,axis)
            #print(np.dot(vector,rotation_matrix))
            return (np.dot(vector,rotation_matrix))

        else:
            rotation_matrix = determine_rotation_matrices(angle,axis)
            #print(np.dot(vector,rotation_matrix))
            return (np.dot(vector,rotation_matrix))
    
    def create_df_from_list(Data):
        Data = Data.split(sep='\n')
        return pd.DataFrame([[Data[0][0:2], Data[0][3:22], Data[0][22:41], Data[0][41:60], Data[0][60:79]],
                        [0            , Data[1][3:22], Data[1][22:41], Data[1][41:60], Data[1][60:79]],
                        [0            , Data[2][3:22], Data[2][22:41], Data[2][41:60], Data[2][60:79]],
                        [0            , Data[3][3:22], Data[3][22:41], Data[3][41:60], Data[3][60:79]],
                        [0            , Data[4][3:22], Data[4][22:41], Data[4][41:60], Data[4][60:79]],
                        [0            , Data[5][3:22], Data[5][22:41], Data[5][41:60], Data[5][60:79]],
                        [0            , Data[6][3:22], Data[6][22:41], Data[6][41:60], Data[6][60:79]],
                        [0            , Data[7][3:22], Data[7][22:41], Data[7][41:60], Data[7][60:79]]])

    def remove_D(string):
        E = int(string[16:19])
        number = float(string[0:15])

        if E > 0:
            E = E
        elif E < 0:
            E = E-1
        elif E == 0:
            return number
        
        return number*(10**E)
    
    df = create_df_from_list(brd)

    prn = df.loc[0][0]
    epoch= eph

    crs = remove_D(df.loc[1][2])
    delta_n = remove_D(df.loc[1][3])
    m0= remove_D(df.loc[1][4])

    cuc= remove_D(df.loc[2][1])
    e= remove_D(df.loc[2][2])
    cus= remove_D(df.loc[2][3])
    sqrt_a= remove_D(df.loc[2][4])

    Toe= remove_D(df.loc[3][1])
    cic= remove_D(df.loc[3][2])
    omega_0= remove_D(df.loc[3][3])
    cis= remove_D(df.loc[3][4])

    i0= remove_D(df.loc[4][1])
    crc= remove_D(df.loc[4][2])
    omega= remove_D(df.loc[4][3])
    omega_dt= remove_D(df.loc[4][4])

    idt= remove_D(df.loc[5][1])

    reference_epoch = Toe - (86400 * 3)
    calculation_epoch = epoch
    tk = calculation_epoch - reference_epoch

    Mk = m0 + ((math.sqrt(u)/ sqrt_a**3) + delta_n)*tk

    def Ek(theta):
        return Mk + e*math.sin(theta)

    def solve_iteratively(previous_value, f, tolerance=1e-15, max_iterations=1000):
        current_value = f(previous_value)
        iterations = 1

        while abs(current_value - previous_value) > tolerance and iterations < max_iterations:
            previous_value = current_value
            current_value = f(previous_value)
            iterations += 1

        if iterations == max_iterations:
            print(f"Maximum iterations ({max_iterations}) reached.")

        return current_value
    
    Ek = solve_iteratively(Mk + e*math.sin(Mk),Ek)
    v = 2*np.arctan( np.sqrt((1+e)/(1-e)) / np.tan(Ek/2.0))
    #v=Ek
    r = a*(1-e*np.cos(Ek))
    Uk = omega + v + (cuc*math.cos(2*(omega+v))) + (cus*math.sin(2*(omega+v)))
    rk = a*(1-(e)*math.cos(Ek)) + (crc)*math.cos(2*((omega)+v)) + (crs)*math.sin(2*((omega)+v))
    ik = (i0) + (idt)*tk + (cic)*math.cos(2*((omega)+v)) + (cis)*math.sin(2*((omega)+v))
    lambda_lon_asc = (omega_0) + ((omega_dt)-wE)*tk - wE*(Toe)

    vector = np.array([rk,0,0])
    aft_z_rot = rotation(vector,-Uk,3)
    #print(aft_z_rot)

    aft_x_rot = rotation(aft_z_rot,-ik,1)
    #print(aft_x_rot)

    aft_z_rot1 = rotation(aft_x_rot,math.radians(-lambda_lon_asc),3)

    print(aft_z_rot1)
    #print(Uk,rk,ik,lambda_lon_asc,v,Ek,Mk)

cal_brd(eph=epoch,brd=Data)

from io import StringIO

DATA = StringIO('''8 15  0.00000000 , -2476.397097 , -15103.929148 , 21848.727349
8 30  0.00000000 ,  -284.942068 , -16146.758222 , 21200.065741
8 45  0.00000000 ,  1768.238636 , -17262.647720 , 20190.304478
9 0  0.00000000 ,  3649.599963 , -18416.066775 , 18835.571501
9 15  0.00000000 ,  5332.551691 , -19566.885899 , 17157.968770 
9 30  0.00000000 ,  6798.328168 , -20671.870562 , 15185.275708
9 45  0.00000000 ,  8036.565166 , -21686.324498 , 12950.547633
10 0  0.00000000,  9045.556818 , -22565.822001 , 10491.610607
10 15  0.00000000,  9832.177338 , -23267.962937 ,  7850.456596
10 30  0.00000000,  10411.465187,  -23754.081020,  5072.545750''')

def cal_sp3(eph, sp3):
    def lagrange_interp(x, y, xi):

        n = len(x)
        w = np.zeros(n)
        yi = 0.0

        for i in range(n):
            w[i] = np.prod([(xi - x[j])/(x[i] - x[j]) for j in range(n) if j != i])
            yi += w[i]*y[i]

        return yi

    def convert_to_daysec(data,item):
        hour = float(data['epoch'][item].split(sep=' ')[0]) * 60 *60
        minute = float(data['epoch'][item].split(sep=' ')[1]) * 60 
        second = float(data['epoch'][item].split(sep=' ')[3])

        #print(hour,minute,second)
        return hour + minute + second

    df = pd.read_csv(sp3, sep =",", names=['epoch','X','Y','Z'])

    for item in range(len(df)):
        df['epoch'][item] = convert_to_daysec(df,item)

    X_LG = lagrange_interp(df['epoch'],df['X'],eph)
    Y_LG = lagrange_interp(df['epoch'],df['Y'],eph)
    Z_LG = lagrange_interp(df['epoch'],df['Z'],eph)

    print(X_LG,Y_LG,Z_LG)

cal_sp3(eph=epoch,sp3=DATA)