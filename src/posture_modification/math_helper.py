import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

from constants import *


def base_round(x, base):
    return base * round(x/base)
# reorders quaternion cause fuck scipy ig
def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], [1], [2]])

def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])

def get_axes(x,y,z):
    return np.stack((norm(x), norm(y), norm(z)))

def norm(v):
    return v / np.linalg.norm(v)

def get_vector(p0, p1, normalize=False):
    # take array of p0 and p1 and compute the vector from p0 to p1
    v = p1 - p0

    if normalize:
        l = np.linalg.norm(v, axis=1)
        v /= l[:, None] # to allow for 2d array math

    return v

def calc_rot_q(v0, v1):
    # compute the quaternion to produce the rotation from v0 to v1
    # arrays of v0 and v
    # quaternion = (cos theta/2, n* sintheta/2) for n is the cross product and theta is the rotation
    cross = np.cross(v0, v1)
    # checking for parallel vectors
    if np.all(np.abs(cross) < np.array([[E,E,E] for _ in range(len(cross))])):
        if np.sum(np.linalg.norm(v1-v0)) < np.sum(np.linalg.norm(v1+v0)):
            # two vectors are very close to each other
            # return identity quaternion
            return np.array([[1,0,0,0] for _ in range(len(v0))])
        else:
            # two vectors are opposite each other so 180 degree rotation
            return np.array([[0,0,1,0] for _ in range(len(v0))])

    cross_lens = np.linalg.norm(cross, axis=1)
    n = cross / cross_lens[:, None]
    dots = np.sum(v0 * v1, axis=1)
    thetas = np.arctan2(cross_lens,dots)
    q = np.vstack((np.cos(thetas/2), *((np.sin(thetas/2)[:, None]*n).T))).T

    return q

def q_to_local(glob_quat, glob_m, loc_m):
    # computes the rotation quaternion given in global orientation and returns rotation in local orientation
    # assuming global quaternion is in w,x,y,z
    glob_quat = glob_quat[1], glob_quat[2], glob_quat[3], glob_quat[0] # scipy takes it in xyzw
    glob_quat_rotm = R.from_quat(glob_quat).as_matrix()
    dcm = np.dot(glob_m, loc_m) #rotation matrix from global to local

    local_quat_m = np.dot(dcm, np.dot(glob_quat_rotm, np.linalg.inv(dcm))) #dcm * quat_rotation_matrix * dcm^-1

    local_quat = R.from_matrix(local_quat_m).as_quat()
    local_quat = np.array([local_quat[3], local_quat[0], local_quat[1], local_quat[2]]) # reorder back to wxyz

    # if local_quat[0] < 1e-15:
    #     local_quat[0] = 0
    return local_quat

    # w,x,y,z
    return q

# filtering

def lp_filter(x, sample_rate, cutoff):
    lp_order = 3
    fs = sample_rate
    nyq = 0.5*fs
    cutoff = cutoff
    crit_freq = cutoff/nyq
    b,a= butter(lp_order,crit_freq,'lowpass')
    return filtfilt(b,a,x)

# randomizations for synthetic legs

# generates a set of random numbers that follow each other on gaussian delay
def gaussian_intervals(time_range, loc, scale):
    times = []

    t = time_range[0]

    while t < time_range[1]:
        times.append(t)
        t += np.random.normal(size=1,loc=loc, scale=scale)[0]

    times.append(time_range[-1]-1)

    return times

def ranged_gaussian(r, scale):
    loc = (r[1] + r[0])/2

    val = np.random.normal(size=1, loc=loc, scale=np.abs(scale))[0]

    return np.clip(val, r[0], r[1]) # clamps to range

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]



def rand_time_var_vectors(base_vector, ranges, n_frames):
    # ranges in degrees
    intervals = np.ceil(gaussian_intervals((1, n_frames), n_frames/4, n_frames/32))

    d = []
    d.append([(r[1] + r[0])/2 for r in ranges]) # creates the first frame of rotation
    prev = d[0]
    for i in range(1, n_frames):
        if i not in intervals: # adds empty rows for each missing frame
            d.append([np.nan, np.nan, np.nan])
            continue


        # only works per interval

        # gaussian noise sampling
        # timestep = []
        # for ax in range(3):
        #     # generates some maxes and mins for an up dwon curve
        #     r = ranges[ax]
        #     if i % 2 == 0:
        #         r = (prev[ax], r[1]) # go from current value up to max
        #     elif i % 2 == 1:
        #         r = (r[0], prev[ax])
        #     loc = np.abs(r[1] - r[0]) / 8

        #     timestep.append(ranged_gaussian(r, loc))

        # unfirom sampling
        timestep = []
        for ax in range(3):
            timestep.append(np.random.uniform(ranges[ax][0], ranges[ax][1]))

        prev = timestep
        d.append(timestep)

    d = np.stack(d)
    for ax in range(3):
        nans, x = nan_helper(d[:,ax])
        d[nans,ax] = np.interp(x(nans), x(~nans), d[~nans,ax])

    # generate a 3d array of euler rotations

    vecs =[]
    for rot in d:
        vecs.append(R.from_euler("xyz", rot, degrees=True).apply(base_vector))

    return vecs
