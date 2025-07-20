import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants

import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import matplotlib.patches as patches
import matplotlib.transforms as transforms
from PIL import Image as PIL_Image
import os
import glob

rhoF = 998.2 #density of fluid [kg/m^3]
mu = 0.001003 #dynamic viscosity of fluid [Pa*s]
rhoP = 2700 #density of plate [kg/m^3]
beta = 1/20 #aspect ratio of plate [ -]
L = 0.04 #length of plate [m]
h = L*beta #height of plate [m]
V = L*h #volume of plate [m^2]
m = V*rhoP #mass of plate [kg/m]
mF = V*rhoF #mass of fluid occupied by plate [kg/m]
Iz = 1/12* m*(h**2+L**2) #moment of inertia for plate [kg*m]
IzF = 1/12* mF*(h**2+L**2) #moment of inertia for fluid [kg*m]
dt = 0.00025 #time step size [s]
t_start = 0 #start time [s]
t_end = 3 #end time [s]

alphaDotMatrix = range(0,91,5) #vector giving dalpha /dt simulations
pos = np.array([0, 0]) #initial position of plate [m]
u = np.array([0, -0.0002]) #initial velocity of plate [m/s]

w = np.array([0, 0]) #velocity of fluid surround the plate [m/s]
v = w-u #initial relative fluid velocity [m/s]
theta = -30* np.pi/180 #initial orientational angle [deg]
omega = 0 #initial angular velocity of plate [rad/s]
domega = 0 #initial angular acceleration of plate [rad/s]
alphaDot = 0 #intial change in angle of attack [rad/s]

c1 = np.pi/2
c2 = np.pi


coefficient_matrix = np.array([
    [0, 0, 0, 0],
    [-357.1757, 86.8499, 0.7730, 0.1468],
    [-339.6383, 124.0477, -7.0129, 0.5075],
    [-321.5198, 151.9937, -15.5799, 1.0821],
    [-305.1881, 175.2423, -24.9560, 1.8921],
    [-273.7018, 183.2955, -32.3156, 2.7558],
    [-231.0180, 175.2214, -35.9001, 3.4559],
    [-186.1378, 156.3123, -35.7169, 3.8684],
    [-144.3193, 131.6641, -32.4764, 3.9553],
    [-109.2020, 106.3660, -27.4888, 3.7768],
    [-80.9141, 82.6345, -21.5865, 3.3881],
    [-65.3751, 70.8033, -19.5711, 3.4626],
    [-50.9844, 56.5638, -15.2833, 3.1031],
    [-44.3648, 51.5779, -14.8134, 3.2965],
    [-33.3199, 37.1812, -8.5775, 2.4114],
    [-9.1105, -3.2495, 13.9542, -1.7400],
    [-0.9539, -17.3068, 22.3751, -3.4478],
    [7.8310, -33.9494, 33.2012, -5.8047],
    [15.8961, -50.5052, 44.8213, -8.5169]
])

alpha_start_matrix = np.array([
    0, 1.6, 3.2, 9.0, 7.5, 9.0, 11.0, 13.0, 14.0, 15.0,
    17.0, 18.0, 18.0, 20.0, 20.0, 21.0, 24.5, 25.6, 25.0
])

alpha_end_matrix = np.array([
    0, 12.0, 15.7, 18.9, 21.8, 24.7, 27.6, 30.6, 33.7, 36.8,
    39.8, 42.7, 45.3, 47.6, 49.7, 52.7, 55.0, 57.8, 61.0
])


t90cw = np.array([
    [0, 1],
    [-1, 0]
]) 

t90ccw = np.array([
    [0, -1],
    [1, 0]
]) 


def Re_get(v):
    return np.linalg.norm(v)*L*rhoF/mu


def dragCoeff0_get(Reynold):
    return 0.023+5.45/(Reynold**0.58-0.80)

def dragCoeff90_get(Reynold):
    if Reynold <= 5:
        return 1.75 + 5.0/(0.20*Reynold**1.18)
    elif Reynold <= 75:
        return 1.75 + 5.0/(0.20*Reynold**1.18)
    elif Reynold <= 1280:
        return 1.63*10**-9 * Reynold**3 - 5.17*10**-6 * Reynold**2 + 5.41*10**-3 * Reynold + 1.54
    elif Reynold <= 20000:
        return 3.05 + 5.0/(0.045*Reynold**0.80)
    else:
        return 3.05 + 5.0/(0.045*Reynold**0.80)

def dragCoeff_unsteady_get(alpha, alphaDot, Reynold):
    if alphaDot < 0.0002:
        return dragCoeff_steady_get(alpha, Reynold)
    
    a2 = np.pi / (2*np.abs(alphaDot))
    a1 = a2**0.5
    dragCoeff0 = dragCoeff0_get(Reynold)
    dragCoeff90 = dragCoeff90_get(Reynold)

    dragCoeff_unsteady = np.pi/a2 + dragCoeff0 - dragCoeff0/a2 * np.cos((a1*abs(alpha)+np.pi/2))**2 - dragCoeff90/a2 * np.sin((a1*abs(alpha)+np.pi/2))**6

    return dragCoeff_unsteady

def dragCoeff_steady_get(alpha, Reynold):
    dragCoeff0 = dragCoeff0_get(Reynold)
    dragCoeff90 = dragCoeff90_get(Reynold)

    dragCoeff_steady = dragCoeff0+(dragCoeff90 -dragCoeff0)*(np.sin(np.abs(alpha)))**3

    return dragCoeff_steady


def sign_cp_fcn(theta ,v):
    sign_cp = np.zeros(2)
    phi = np.arctan(v[1]/v[0])*180/np.pi #angle between v and the global x- axis
    theta_scaled = ((theta*180/np.pi) % 180)-180 #scaled to period of 180 deg

    if v[1] > 0:
        if v[0] > 0:
            if theta_scaled > -90 and theta_scaled < -90+ phi:
                sign_cp[0] = 1
            else:
                sign_cp[0] = -1
        else:
            if theta_scaled < -90 and theta_scaled > -90 + phi:
                sign_cp[0] = -1
            else:
                sign_cp[0] = 1 
    elif v[1] < 0:
        if v[0] > 0:
            if theta_scaled < -90 and theta_scaled > -90 + phi:
                sign_cp[0] = 1
            else:
                sign_cp[0] = -1
        else:
            if theta_scaled > -90 and theta_scaled < -90+ phi:
                sign_cp[0] = -1
            else:
                sign_cp[0] = 1
    
    if v[1] > 0:
        if v[0] > 0:
            if theta_scaled < 0 and theta_scaled > -90+ phi:
                sign_cp [1] = 1
            else:
                sign_cp [1] = -1
        else:
            if theta_scaled < 0 and theta_scaled > -90+ phi:
                sign_cp [1] = -1
            else:
                sign_cp [1] = 1
    elif v[1] < 0:
        if v[0] > 0:
            if theta_scaled > -90+ phi and theta_scaled < 0:
                sign_cp [1] = 1
            else:
                sign_cp [1] = -1
        else:
            if theta_scaled > -90+ phi and theta_scaled < 0:
                sign_cp [1] = -1
            else:
                sign_cp [1] = 1

    return sign_cp


num_iter = int((t_end-t_start)/dt)+1

pos_vec = np.zeros((num_iter, 2)) #position
u_vec  = np.zeros((num_iter, 2)) #trans . velocity
u_mag_vec = np.zeros((num_iter, 1)) #trans . velocity magnitude
a_vec  = np.zeros((num_iter, 2)) #trans . acceleration
a_vec_mag  = np.zeros((num_iter, 1)) #trans . acceleration magnitude
theta_vec = np.zeros((num_iter, 1)) #orientation angle
alpha_vec = np.zeros((num_iter, 1)) #angle of attack
omega_vec = np.zeros((num_iter, 1)) #angular velocity
domega_vec = np.zeros((num_iter, 1)) #angular acceleration
Fd_vec = np.zeros((num_iter, 2)) #drag force
Fl_vec = np.zeros((num_iter, 2)) #lift force
cp_abs_vec = np.zeros((num_iter, 1)) #centre of pressure magnitude
t_vec = np.zeros((num_iter, 1)) #time


phi = np.arctan2(v[1], v[0])
alpha = 2*c1/np.pi * np.arctan(np.pi/c2 * 1/np.tan(theta - np.pi/2 - phi))

i = 0
t = t_start


while t < t_end:

    tToLocal = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    tToGlobal = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]) 

    v = w-u

    #Reynold
    Reynold = Re_get(v)

    #drag coefficient
    if abs(alphaDot) > np.pi/2:
        alphaDot = np.pi/2

    dragCoeffUnsteady = dragCoeff_unsteady_get(alpha, alphaDot, Reynold)
    dragCoeffSteady = dragCoeff_steady_get(alpha, Reynold)

    if dragCoeffSteady < dragCoeffUnsteady:
        dragCoeff = dragCoeffUnsteady
    else:
        dragCoeff = dragCoeffSteady

    #lift coefficient
    alphaDotRounded = np.round(abs(alphaDot)*180/np.pi/5)*5

    if alphaDotRounded > 90:
        alphaDotRounded = 90

    alphaDotIndex = alphaDotMatrix.index(alphaDotRounded)

    if abs(alpha) < alpha_start_matrix[alphaDotIndex]*np.pi/180: 
        liftCoeff = abs(alpha)*180/np.pi*0.1263 #linear regression
    elif abs(alpha) < alpha_end_matrix[alphaDotIndex]*np.pi/18:
        liftCoeff = coefficient_matrix[alphaDotIndex][0]*abs(alpha)**3 + coefficient_matrix[alphaDotIndex][1]*abs(alpha)**2 + coefficient_matrix[alphaDotIndex][2]*abs(alpha) + coefficient_matrix[alphaDotIndex][3] #dalpha /dt dependent CL
    else:
        liftCoeff = 7.0763*abs(alpha)**5 - 28.9422*abs(alpha)**4 + 41.3903*np.abs(alpha)**3 - 25.7446*abs(alpha)**2 + 7.3899*abs(alpha) + 0.0375 

    #
    buoyancyF = np.array([0, V* rhoF *constants.g]) #buoyancy force vector
    gravityF = np.array([0, -constants.g*m]) #gravity force vector
    
    dragFAbs = dragCoeff *1/2* rhoF *L* np.linalg.norm(v)**2 #absolute drag force
    liftFAbs = liftCoeff *1/2* rhoF *L* np.linalg.norm(v)**2 #absolute lift force

    dragF = v/np.linalg.norm(v) * dragFAbs #drag force vector
    if alpha < 0:
        liftF = t90cw@(v/np.linalg.norm(v) * liftFAbs) #turn lift clock - wise
    else:
        liftF = t90ccw@(v/np.linalg.norm(v) * liftFAbs) #turn lift counter -close - wise

    #C = np.array([[0.0731 , 0], [0, 17.3]]) #added mass coefficient matrix
    C = np.array([[0, 0], [0, 0]]) #added mass coefficient matrix
    Ma = C*rhoF*V #added mass matrix
    
    M = np.identity(2)*m #total mass matrix
    a_local = np.linalg.solve(M+Ma, tToLocal@(dragF + liftF + gravityF + buoyancyF)) #accele . in local coordinates
    a_global = tToGlobal@a_local #accele . in global coordinates
    a = a_global #update acceleration

        
    
    u = u+a*dt #update translational velocity
    v = w-u #update relative fluid velocity
    dpos = u*dt + 1/2*a*dt**2 #find change in position
    pos = pos + dpos #update position
    phi = np.arctan2(v[1], v[0])
    alpha, alphaOld = 2 * c1 / np.pi * np.arctan(1 / np.tan(np.pi/c2*theta - np.pi/2 - phi)), alpha
    alphaDot = (alpha - alphaOld)/dt
    
    cpAbs = L*0.015*(1 - np.sin(abs(alpha))**3) #find centre of pressure dist .

    signCp = sign_cp_fcn(theta, v) #call fun. to find sign of cp
    cp = [cpAbs*abs(np.cos(theta))*signCp[0], cpAbs*abs(np.sin(theta) )*signCp[1]] #find cp vector

    T_offset = np.linalg.det([cp, dragF+liftF]) #calc . torque aerodyn . forces

    DRCoeff = 2
    T_resist = -rhoF*DRCoeff*omega*abs(omega)*(1/4)*(L/2)**4 #calculate torque resistance

    #C66 = 7.03
    C66 = 0

    M66 = C66*IzF

    T_total = T_offset + T_resist #total torque on plate
    domega = T_total /(Iz+M66) #find angular acceleration
    dtheta = omega*dt + 1/2*domega*dt**2 ##find change in orient . angle
    omega = omega + domega*dt #update angular velocity
    theta = theta + dtheta #update orientation angle

    pos_vec[i] = pos #position
    u_vec[i] = u #trans . velocity
    u_mag_vec[i] = np.linalg.norm(u) #trans . velocity magnitude
    a_vec[i] = a #trans . acceleration
    a_vec_mag[i] = np.linalg.norm(a) #trans . acceleration magnitude
    theta_vec[i] = theta #orientation angle
    alpha_vec[i] = alpha #angle of attack
    omega_vec[i] = omega #angular velocity
    domega_vec[i] = domega #angular acceleration
    Fd_vec[i] = dragF #drag force
    Fl_vec[i] = liftF #lift force
    cp_abs_vec[i] = cpAbs ;#centre of pressure magnitude
    t_vec[i] = t #time

    t += dt
    i += 1


xmax, xmin = max(pos_vec[:,0]/L+1), min(pos_vec[:,0]/L-1) 
ymax, ymin = max(pos_vec[:,1]/L+1), min(pos_vec[:,1]/L-1)
 
plt.plot(pos_vec[:,0]/L, pos_vec[:,1]/L)     # The marker shows the path points
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Path on Coordinate Plane')
plt.grid(True)                 # Adds the grid

plt.show()

plt.plot(t_vec, theta_vec)     # The marker shows the path points
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Path on Coordinate Plane')
plt.grid(True)                 # Adds the grid

plt.show()


mpl.use('Agg')

num_frames = len(t_vec)//50



# Setup plot
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-') # Added color for visibility
ax.set(xlabel='X-axis', ylabel='Y-axis', title='Animated 2D Path')
ax.grid(True)
# Set axis limits to prevent them from changing dynamically
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Animation function
def update(frame):
    line.set_data(pos_vec[:(frame*50), 0] / L, pos_vec[:(frame*50), 1] / L)
    # Return the artist as a sequence (a single-element tuple)
    return (line,)

iteration = 0

while True:
    animation_dir = 'fall'+str(iteration)+'.gif'
    if os.path.exists(animation_dir):
        iteration+=1
    else:
        break


# Create, save, and display the animation
aniSimp_dir = 'fall'+str(iteration)+'.gif'

ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
HTML(ani.to_jshtml())

ani.save(aniSimp_dir, writer='pillow', fps=60)
#Image(filename='fall.gif')



frames_dir = "animation_frames"
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

sampling_step = 50
if len(pos_vec) < sampling_step:
    raise ValueError("Not enough data points for the current sampling step.")
frames_count = len(pos_vec) // sampling_step

print(f"Generating {frames_count} frames...")

# -- Step 1: Generate and Save Each Frame --
for frame_num in range(frames_count):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title("Animated Rectangle")

    rect = patches.Rectangle((0, 0), 1.0, 0.05, facecolor='skyblue', edgecolor='black')
    ax.add_patch(rect)

    idx = frame_num * sampling_step
    current_x, current_y = pos_vec[idx]/L
    current_angle_rad = theta_vec[idx] # The angle is in radians

    # --- THE FIX: Convert angle from radians to degrees ---
    current_angle_deg = np.rad2deg(current_angle_rad)
    
    # Position and rotate the rectangle using the angle in degrees
    transform = transforms.Affine2D().rotate_deg_around(current_x, current_y, current_angle_deg)
    rect.set_xy((current_x - 1.0 / 2, current_y - 0.05 / 2))
    rect.set_transform(transform + ax.transData)

    filename = os.path.join(frames_dir, f"frame_{frame_num:04d}.png")
    plt.savefig(filename)
    plt.close(fig)

print("All frames generated successfully.")

# -- Step 2: Compile Frames into a GIF --
print("Compiling frames into 'final_animation.gif'...")

frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
if not frame_files:
    raise ValueError("No frames were generated to compile.")

frames = [PIL_Image.open(image) for image in frame_files]
frame_one = frames[0]

ani_dir = 'final_animation'+str(iteration)+'.gif'

frame_one.save(
    ani_dir,
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=40,
    loop=0
)

print("GIF saved successfully.")

# -- Step 3: Display the final GIF --
#Image(filename=ani_dir)
