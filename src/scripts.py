import numpy as np

def calc_distance(r1, r2, side_length=[10.0, 10.0, 10.0]):
      #manage periodic boundary 
      r12_ = [0.0,0.0,0.0]
      r12_[0] = (r1[0] - r2[0] + side_length[0]/2.) % side_length[0] - side_length[0]/2.
      r12_[1] = (r1[1] - r2[1] + side_length[1]/2.) % side_length[1] - side_length[1]/2.
      r12_[2] = (r1[2] - r2[2] + side_length[2]/2.) % side_length[2] - side_length[2]/2.
      return np.linalg.norm(r12_)


      