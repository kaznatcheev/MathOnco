import FluorescentArea as fa
import os
import time
import numpy as np

def ConvertArchive(root_dir, subdir_match, col_let_array, row_num_array):
    red_list_all = []
    green_list_all = []
    
    for dirName, subdirList, fileList in os.walk(rootDir):
        if len(subdirList) > 0:
            if subdirList[0] == subDirMatch:
                #track time
                start_time = time.clock()

                #create slices for this timestep
                red_slice = np.zeros((len(col_let_array),len(row_num_array)))
                green_slice = np.zeros((len(col_let_array),len(row_num_array)))

                for col_let_num in range(len(col_let_array)):
                    col_let = col_let_array[col_let_num]
                    for row_num_num in range(len(row_num_array)):
                        row_num = row_num_array[row_num_num]
						
						#analyze each of the three images
                        nums_1, im = fa.AreaCount(col_let,row_num,1,
							dirName+'/'+subDirMatch+'/')
                        cv2.imwrite(dirName + '/'+subDirMatch+ \
							'/yol_'+col_let+str(row_num)+'-1.jpg',im)

                        nums_2, im = fa.AreaCount(col_let,row_num,2,
							dirName+'/'+subDirMatch+'/')
                        cv2.imwrite(dirName + '/'+ subDirMatch+ \
							'/yol_'+col_let+str(row_num)+'-2.jpg',im)

                        nums_3, im = fa.AreaCount(col_let,row_num,3,
							dirName+'/'+subDirMatch+'/')
                        cv2.imwrite(dirName + '/'+subDirMatch+ \
							'/yol_'+col_let+str(row_num)+'-3.jpg',im)

                        red_slice[col_let_num,row_num_num] \
							= nums_1[0] + nums_2[0] + nums_3[0]
                        green_slice[col_let_num,row_num_num] \
							= nums_1[1] + nums_2[1] + nums_3[1]

                red_list_all.append(red_slice)
                green_list_all.append(green_slice)

                end_time = time.clock()

                print(dirName + ' took ' \
					+ str(end_time - start_time) + ' seconds')
	
	return red_list_all, green_list_all

import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize
	
def AlectinibPlot(r_1,g_1, y_len, step_size = 1, pwl = True ):
    matplotlib.rcParams['figure.figsize'] = (20, 10)
	
	if pwl: #if we are fitting piecewise linear then define the model
		def piecewise_linear(x, x0, y0, k1, k2):
			return np.piecewise(x, [x < x0], 
				[lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
		red_rates = []
		green_rates = []
	else: #if fitting linear rates, then initialize output array
		red_rates = np.zeros((y_len,2))
		green_rates = np.zeros((y_len,2))
	
	#initialize our arrays of plot axes
    f, plot_arr = plt.subplots(2,y_len,sharex='all', sharey='all')
    plt.yscale('log')
	
    for y in range(y_len):
        for offset in range(2):
			
			#create all_t to collect data from all 3 replicates
            x_all_t = []
            r_all_t = []
            g_all_t = []
			
            for i in range(3):
                x_all_t = np.append(x_all_t,
					range(len(r_1[::step_size,offset*3 + i,y])))
                r_all_t = np.append(r_all_t,r_1[::step_size,offset*3 + i,y])
                g_all_t = np.append(g_all_t,g_1[::step_size,offset*3 + i,y])
				
				#plot the red and green dynamics
                plot_arr[offset,y].plot(range(len(r_1[::step_size,offset*3 + i,y])),
					r_1[::step_size,offset*3 + i,y],'r-',
                    range(len(g_1[::step_size,offset*3 + i,y])),
					g_1[::step_size,offset*3  + i,y],'g-')

            if pwl: #fit piecewise linear of two lines
                r_p , r_e = optimize.curve_fit(piecewise_linear, x_all_t, 
					np.log(r_all_t))
                g_p , g_e = optimize.curve_fit(piecewise_linear, x_all_t, 
					np.log(g_all_t))
				
				#plot the lines of best fit
                xd = np.linspace(0, len(r_1[::step_size,0,y]), num=100)
                plot_arr[offset,y].plot(xd,np.exp(piecewise_linear(xd,*r_p)),'k-', 
					xd,np.exp(piecewise_linear(xd,*g_p)),'k-')
            else: #calculate the single line of best fit:
                [ra,rb] = np.polyfit(x_all_t,np.log(r_all_t),1)
                [ga,gb] = np.polyfit(x_all_t,np.log(g_all_t),1)
				
				#record the growth rates
				red_rates[y,offset] = ra
				green_rates[y,offset] = rb
				
				#figure out the line starting and ending points
                r_start = np.exp(rb)
                r_stop = np.exp(ra*len(r_1[::step_size,0,y]) + rb)

                g_start = np.exp(gb)
                g_stop = np.exp(ga*len(g_1[::step_size,0,y]) + gb)
                
				#plot the lines from their start/end points
                plot_arr[offset,y].plot([0,len(r_1[:,0,y])],[r_start,r_stop],'k-', 
					[0,len(g_1[:,0,y])],[g_start,g_stop],'k-')
	
	return red_rates, green_rates
