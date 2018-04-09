import numpy as np
import math
from operator import itemgetter
from base_algorithm import BaseAlgorithm


class Direct(BaseAlgorithm):
    
    def __init__(self, function_wrapper, number_of_variables = 1, objective = "maximization"):
        super().__init__(function_wrapper,number_of_variables,objective)
    

    def search(self, iterations = 10):

       # number of iterations
       self.iterations = iterations
       
       # global/local weight parameter
       epsilon = 10**-4
              
       nFunc = 0
       
     
       x_L = np.matrix(self.function_wrapper.minimum_decision_variable_values())
       x_U = np.matrix(self.function_wrapper.maximum_decision_variable_values())
    
      
       x_L = x_L.T
       x_U = x_U.T
       
       # convergence tolerance
       tolle2 = 10**-12
       
       # Step 1: Initialization, set first point to center of the unit hypercube
       m = 1 #current number of rectangles
       # matrix with all rectangle centerpoints
       C = np.asmatrix(np.ones([self.number_of_variables,1]))/2 
      # transform C to original search space
       x_m = x_L + np.multiply(C,(x_U - x_L))
       # calculate the first f_min
       if self.objective == 'minimization':
           f_min = self.function_wrapper.objective_function_value(x_m)
       else:
           # if objective = "maximization" evaluate 1/f(x)
           # max f(x) -> min -f(x)
           f_min = -1*self.function_wrapper.objective_function_value(x_m)
       
       nFunc = nFunc + 1
       i_min = 1
       # Matrix with all rectangle side lengths in each dimension
       L = np.asmatrix(np.ones([self.number_of_variables,1]))/2
       # Vector with distances from centerpoint to the vertices
       D = math.sqrt(sum(np.square(L)))
       # Vector with function values
       if type(f_min) == np.matrixlib.defmatrix.matrix:
          f_min = np.asscalar(f_min)
          F = [f_min]
       else:
          F = [f_min]
          
       D = np.asmatrix([D]).tolist()
       # Row vector of all different distances
       d = D
       # Row vector of minimum function value for each distance
       d_min = [f_min]
       
       # Iteratio loop
       for t in range(1,iterations + 1): 
           # Step 2 Identify the set S of all potentially optimal rectangles
           S = [] # set of all potentially optimal rectangles
           S_1 = [] 
           idx = d.index(D[i_min-1])
           
           for i in range(idx+1,len(d)+1):
               D_index = [k for k, x in enumerate(D) if x == d[i-1]]
               F_index = [k for k, x in enumerate(F) if x == d_min[i-1]]
               idx2 = np.array(sorted(list(set(D_index)&set(F_index))))
               for kk in range(len(idx2)):
                   S_1.append(idx2[kk])
           # S_1 now includes all rectangles i, with D(i) >= D(i_min)
           # F(i) is the minimum function value for the current distance.
           
           # Pick out all rectangles in S_1 which lies below the line passing through
           # the points: ( D(i_min), F(i_min) ) and the lower rightmost point.
           S_2 = []
           
           if len(d)-idx-1 > 1:
               a1 = D[i_min-1]
               b1 = F[i_min-1]
               a2 = d[len(d)-1]
               b2 = d_min[len(d)-1]
               # The line is defined by: y = slope*x + const
               slope = (b2-b1)/(a2-a1)
               const = b1 - slope*a1
               
               for i in range(0,len(S_1)):
                   j = np.asscalar(np.array(S_1[i]))
                   if F[j] <= slope*D[j] + const + tolle2:
                      S_2.append(j)
               # S_2 now contains all points in S_1 which lies on or below the line
               # Find the points on the convex hull defined by the points in S_2
               xx = []
               yy = []
               S_3 = [] 
               for kk in range(len(S_2)):
                   xx.append(D[S_2[kk]])
                   yy.append(F[S_2[kk]])
     
               h = self.__conhull(xx,yy) # conhull is an internal subfunction
               
               for i in range(0,len(h)):
                   S_3.append(S_2[int(h[i])-1])
           else:
               S_3 = S_1
           
           S = S_3
           F = np.asmatrix(F)
           
           # STEP 3, Select any rectangle j in S
           for jj in range(1,len(S)+1): # For each potentially optimal rectangle
                j = S[jj-1]
                
                # Step 4, Determine where to sample within rectangle j and how
                # to divide the rectangle into subrectangles. Update f_min
                # and set m = m + delta_m, where delta_m is the number of new
                # points sampled.
                
                # 4.1 Identify the set I of dimensions with the maximum side length.
                # Let delta equal to 1/3 of this maximum length
                max_L = np.asscalar(np.asarray(max(L[:,j])))
                a = L[:,j].tolist()
                I = np.asmatrix([k for k, x in enumerate(a) if x == [max_L]]).T
                delta = 2*max_L/3
                
                # 4.2 Sample the function at the points c +- delta * e_i for all
                # i in I
                w = []
                e_i = []
                for ii in range(1,len(I)+1): # for each dimension with max side length
                    e_i = []
                    i = np.asscalar(I[ii-1])
                    e_i1 = np.asmatrix(np.zeros((i,1)))
                    e_i2 = np.asmatrix(1)
                    e_i3 = np.asmatrix(np.zeros((self.number_of_variables-i-1,1)))
                    e_i =  np.vstack([e_i1,e_i2,e_i3])
                    
                    c_m1 = C[:,j] + delta*e_i # centerpoint for new rectangle
                    # transform c_m1 to original search space
                    x_m1 = x_L + np.multiply(c_m1,x_U-x_L)
                    
                    # function value at x_m1
                    if self.objective == "minimization":
                        f_m1 = self.function_wrapper.objective_function_value(x_m1)
                    else:
                        f_m1 = -1*self.function_wrapper.objective_function_value(x_m1)
                    
                    f_m1_s = f_m1
                    nFunc = nFunc + 1
                    
                    c_m2 = C[:,j] - delta*e_i # centerpoint for new rectangle
                    # transform c_m2 to original search space
                    x_m2 = x_L + np.multiply(c_m2,x_U-x_L)
                    
                    # function value at x_m2
                    if self.objective == "minimization":
                        f_m2 = self.function_wrapper.objective_function_value(x_m2)
                    else:
                        f_m2 = -1*self.function_wrapper.objective_function_value(x_m2)

                    f_m2_s = f_m2
                    nFunc = nFunc + 1
                    w.append(min(f_m1_s,f_m2_s))
                    # matrix with all rectangle centerpoints
                    C = np.hstack([C,c_m1,c_m2])
                    # vector with function values
                    if type(f_m1) == np.matrixlib.defmatrix.matrix:
                        F = np.hstack([F,f_m1,f_m2])
                    else:
                        F = np.hstack([F,np.asmatrix([f_m1]),np.asmatrix([f_m2])])
                        
                # 4.3 Divide the rectangle containing C(:,j) into thirds along the
                # dimension in I, starting with the dimension with the lowest
                # value of w(ii)
                b,a = zip(*sorted(enumerate(w),key = itemgetter(1)))
    
     
                for ii in range(1,len(I)+1):
                    i = I[b[ii-1]]
                    # index for new rectangle
                    ix1 = m + 2*b[ii-1]
                    # index for new rectangle
                    ix2 = m + 2*b[ii-1]+1
                    n_L = ix1 - int(L.shape[1])
                    if n_L>0:
                        zero_n = np.asmatrix(np.zeros([2,n_L]))
                    L[i,j] = delta/2
                    L_ix1 = L[:,j]
                    L_ix2 = L[:,j]
                    if n_L>0:
                        L = np.hstack([L,zero_n,L_ix1,L_ix2])
                    if n_L == 0:
                        L = np.hstack([L,L_ix1,L_ix2])
                    if n_L < 0:
                        L[:,ix1] = L[:,j]
                        L[:,ix2] = L[:,j]
             
                    D[j] = math.sqrt(sum(np.square(L[:,j])))
                    if n_L>0:
                        for kk in range(n_L):
                            D.append(0)
                            D.append(D[j])
                            D.append(D[j])
                    if n_L == 0:
                        D.append(D[j])
                        D.append(D[j])
                    if n_L < 0:
                        D[ix1] = D[j]
                        D[ix2] = D[j]
           
                m = m + 2*len(I)
                
           # Update:   
           f_min = F.min()
           E = max(epsilon*abs(f_min),10**-8)
           dummy_min = np.divide(F-f_min+E,D)
           i_min = dummy_min.argmin()+1
           d = D
           i = 1
           while 1:
                d_tmp = d[i-1]
                idx = [k for k, x in enumerate(d) if x!=d_tmp]
                len_idx = len(idx)
                d_idx = []
                for z in range(1,len_idx+1):
                    d_idx.append(d[idx[z-1]])
                d_idx[0:0] = [d_tmp]
                d = d_idx
                if i == len(d):
                    break
                else:
                    i = i+1
                        
           d.sort()
           d_min = []
           v = []
            
           for i in range(1,len(d)+1):
                v = [] 
                idx1 = [k for k, x in enumerate(D) if x==d[i-1]]
                u = np.array(F).flatten().tolist()
                for z in range(1,len(idx1)+1):
                    v.append(u[idx1[z-1]])
                d_min.append(min(v))

           F = u = np.array(F).flatten().tolist()
           
           decision_variable_values = []
           CC = []
           
      # Transform to original coordinates
       for i in range(m):
            CC.append(np.asscalar(np.array(x_L + np.multiply(C[:,i],(x_U - x_L)))))
      # find all points i with F(i) == f_min
       idx3 = np.asscalar(np.asarray([k for k, x in enumerate(F) if x == f_min]))
      # All points i with F(i) == f_min
       decision_variable_values = [CC[idx3]]
         
           
       if self.objective == 'minimization':
            decision_variable_values = [CC[idx3]]
            f_min_max = f_min
       else:
            decision_variable_values = [CC[idx3]]
            f_min_max = -1*f_min
           
       return { "best_decision_variable_values": decision_variable_values, "best_objective_function_value":f_min_max}
          
      
    def __next1(self,v,m):
        if v == m:
            i = 1
        else:
            i = v+1
        return i

    def __pred(self,v,m):
        if v == 1:
            i = m
        else:
            i = v - 1
        return i

    def __conhull(self,x,y):
        # conhull returns all points on the convex hull, even redundant ones
        # conhull is based on the algorithm GRAHAMSHULL pages 108-109
        # in "Computational Geometry" by Franco P. Preparata and
        # Michael Ian Shamos
        
        x = np.matrix(x).T
        y = np.matrix(y).T
        m = len(x)
        if len(x)!=len(y):
            print('Input dimension must agree, error in conhull-gblSolve')
            return
        if m == 2:
            h = [1,2]
            return h
        if m == 1:
            h = [1]
            return h
        START = 1
        v = START
        w = len(x)
        flag = 0
        h = np.asmatrix(range(1,len(x)+1)).T # Index vector for points in convex hull
        while (self.__next1(v,m) != START) or (flag == 0):
            if self.__next1(v,m) == w:
                flag = 1
            a = v
            b = self.__next1(v,m)
            c = self.__next1(self.__next1(v,m),m)
            D1 = []
            D2 = []
            D3 = []
            D1.append(np.asscalar(np.asarray([np.array(x).flatten().tolist()[a-1]])))
            D1.append(np.asscalar(np.asarray([np.array(y).flatten().tolist()[a-1]])))
            D1.append(1)
            D2.append(np.asscalar(np.asarray([np.array(x).flatten().tolist()[b-1]])))
            D2.append(np.asscalar(np.asarray([np.array(y).flatten().tolist()[b-1]])))
            D2.append(1)
            D3.append(np.asscalar(np.asarray([np.array(x).flatten().tolist()[c-1]])))
            D3.append(np.asscalar(np.asarray([np.array(y).flatten().tolist()[c-1]])))
            D3.append(1)
            Det = np.vstack([np.asmatrix(D1),np.asmatrix(D2),np.asmatrix(D3)])
            if np.linalg.det(Det) >= 0:
                leftturn = 1
            else:
                leftturn = 0
            if leftturn == 1:
                v = self.__next1(v,m)
            else:
                j = self.__next1(v,m)
            
                x1 = []
                y1 = []
                h1 = []
                x2 = []
                y2 = []
                h2 = []
            
                for ll in range(j-1):
                    x1.append(np.asscalar(np.asarray([np.array(x).flatten().tolist()[ll]])))
                    y1.append(np.asscalar(np.asarray([np.array(y).flatten().tolist()[ll]])))
                    h1.append(np.asscalar(np.asarray([np.array(h).flatten().tolist()[ll]])))
                for mm in range(j,m):
                    x2.append(np.asscalar(np.asarray([np.array(x).flatten().tolist()[mm]])))
                    y2.append(np.asscalar(np.asarray([np.array(y).flatten().tolist()[mm]])))
                    h2.append(np.asscalar(np.asarray([np.array(h).flatten().tolist()[mm]])))
                x = np.hstack([np.asmatrix(x1),np.asmatrix(x2)])
                y = np.hstack([np.asmatrix(y1),np.asmatrix(y2)])
                h = np.hstack([np.asmatrix(h1),np.asmatrix(h2)])
                m = m - 1
                w = w - 1
                v = self.__pred(v,m)
        
        h = np.array(h).flatten().tolist()
        return h
        
        
