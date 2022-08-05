# import the necessary packages
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.interpolate import lagrange
from sklearn.metrics import precision_score, recall_score
 
class CAP:
    def __init__(self, actual_results):
        # store the number of bins for the 3D histogram
        self.actual_results = actual_results
    
    def F1_score (self,predicted_result):
        y_values = self.actual_results
        Precision = precision_score(y_values, predicted_result) # (measuring exactness)
        Recall = recall_score(y_values, predicted_result)       # (measuring completeness)
        F1 = 2 * Precision * Recall / (Precision + Recall)  #(compromise between Precision and Recall)
        return F1
 
    
    def ideal(self): # PERFECT MODEL
        num_pos_obs = np.sum(self.actual_results) # Total number of actual 1's in the test data
        num_count = len(self.actual_results)      # Total number of test data
        rate_pos_obs = float(num_pos_obs) / float(num_count) # Actual percentage of 1's in the test data
        ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]}) # Creating perfect model (like we find all 1's first)
        return ideal['x'], ideal['y']
    
    def x_val(self): # RANDOM MODEL
        num_count = len(self.actual_results)      # Total number of test data
        xx = np.arange(num_count) / float(num_count - 1) # Creating Worst/Random model (like straight line)
        return xx
 
    def analyse(self, prediction_probabilities):# INPUT MODEL
        # Initialization
        y_values = self.actual_results
        y_preds_proba = prediction_probabilities[:,1] #store only possibility of the 1 guesses (they stored on the second column)
 
        # PERFECT MODEL CALCULATIONS
        num_pos_obs = np.sum(y_values) # Total number of actual 1's in the test data
        num_count = len(y_values)      # Total number of test data
 
        # INPUT MODEL CALCULATIONS
        y_cap = np.c_[y_values,y_preds_proba] # Puts real test rusults and possibility of the 1 guesses to columns
        y_cap_df_s = pd.DataFrame(data=y_cap) # Prepare them to be plotted
        y_cap_df_s = y_cap_df_s.sort_values(by=[1], ascending=False) # Sorts columns from highest to lovest results in the possibility of the 1 guesses to column (True= lowest to highest)
        y_cap_df_s = y_cap_df_s.reset_index(drop=False) # Re-numarate the raw indexes because they mixed in the previous step
        yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs) # sum each raw of the actual results with it's previous ones and divide each raw to Total number of actual 1's in the test data
        yy = np.append([0], yy[0:num_count-1]) # add the first curve point (0,0) : for xx=0 we have yy=0
        return yy
 
    def fifty_percent(self, prediction_probabilities):
        # INITIAL CALCULATIONS
        y_values = self.actual_results
        y_preds_proba = prediction_probabilities[:,1] #store only possibility of the 1 guesses (they stored on the second column)
        num_pos_obs = np.sum(y_values) # Total number of actual 1's in the test data
        num_count = len(y_values)      # Total number of test data
        xx = np.arange(num_count) / float(num_count - 1) # x-axis values
 
        # INPUT MODEL CALCULATIONS
        y_cap = np.c_[y_values,y_preds_proba] # Puts real test rusults and possibility of the 1 guesses to columns
        y_cap_df_s = pd.DataFrame(data=y_cap) # Prepare them to be plotted
        y_cap_df_s = y_cap_df_s.sort_values(by=[1], ascending=False) # Sorts columns from highest to lovest results in the possibility of the 1 guesses to column (True= lowest to highest)
        y_cap_df_s = y_cap_df_s.reset_index(drop=False) # Re-numarate the raw indexes because they mixed in the previous step
        yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs) # sum each raw of the actual results with it's previous ones and divide each raw to Total number of actual 1's in the test data
        yy = np.append([0], yy[0:num_count-1]) # add the first curve point (0,0) : for xx=0 we have yy=0
 
        # CALCULATE VALUE CORRESPONDS TO 0.5 (50%)(it's one way to calculate the success of the model)
        percent = 0.5  
        row_index = int(np.trunc(num_count * percent)) # trunctes to lower bound  
        if yy[row_index] == yy[row_index+1] or row_index == num_count*percent:
            val = yy[row_index]*1.0
        else:
            x_set = [row_index-3,row_index-2,row_index-1,row_index,row_index+1,row_index+2,row_index+3,row_index+4]
            y_set = [yy[row_index-3],yy[row_index-2],yy[row_index-1],yy[row_index],yy[row_index+1],yy[row_index+2],yy[row_index+3],yy[row_index+4]]
            L = lagrange(x_set,y_set)
            val = L(num_count * percent)
        return val
 
    def area_ratio(self, prediction_probabilities): # CALCULATE THE RATIO OF THE AREAS OF THE BEST MODEL AND INPUT MODEL (anothe way to calculate the success of the model)
        # INITIAL CALCULATIONS
        y_values = self.actual_results
        y_preds_proba = prediction_probabilities[:,1] #store only possibility of the 1 guesses (they stored on the second column)
        num_pos_obs = np.sum(y_values) # Total number of actual 1's in the test data
        num_count = len(y_values)      # Total number of test data
        xx = np.arange(num_count) / float(num_count - 1) # x-axis values
 
        # INPUT MODEL CALCULATIONS
        y_cap = np.c_[y_values,y_preds_proba] # Puts real test rusults and possibility of the 1 guesses to columns
        y_cap_df_s = pd.DataFrame(data=y_cap) # Prepare them to be plotted
        y_cap_df_s = y_cap_df_s.sort_values(by=[1], ascending=False) # Sorts columns from highest to lovest results in the possibility of the 1 guesses to column (True= lowest to highest)
        y_cap_df_s = y_cap_df_s.reset_index(drop=False) # Re-numarate the raw indexes because they mixed in the previous step
        yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs) # sum each raw of the actual results with it's previous ones and divide each raw to Total number of actual 1's in the test data
        yy = np.append([0], yy[0:num_count-1]) # add the first curve point (0,0) : for xx=0 we have yy=0
 
        sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1 # Area under best possible model
        sigma_model = integrate.simps(yy,xx) # Area under worst possible model
        sigma_random = integrate.simps(xx,xx) # Area under inputmodel
        ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random) # calculate ratio
        return ar_value
 
    def analyse_all(self, prediction_probabilities):
        # INITIALIZATION
        y_values = self.actual_results
        y_preds_proba = prediction_probabilities[:,1] #store only possibility of the 1 guesses (they stored on the second column)
 
        # PERFECT MODEL CALCULATIONS
        num_pos_obs = np.sum(y_values) # Total number of actual 1's in the test data
        num_count = len(y_values)      # Total number of test data
        rate_pos_obs = float(num_pos_obs) / float(num_count) # Actual percentage of 1's in the test data
        ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]}) # Creating perfect model (like we find all 1's first)
 
        # RANDOM MODEL CALCULATIONS
        xx = np.arange(num_count) / float(num_count - 1) # Creating Worst/Random model (like straight line)
 
        # INPUT MODEL CALCULATIONS
        y_cap = np.c_[y_values,y_preds_proba] # Puts real test rusults and possibility of the 1 guesses to columns
        y_cap_df_s = pd.DataFrame(data=y_cap) # Prepare them to be plotted
        y_cap_df_s = y_cap_df_s.sort_values(by=[1], ascending=False) # Sorts columns from highest to lovest results in the possibility of the 1 guesses to column (True= lowest to highest)
        y_cap_df_s = y_cap_df_s.reset_index(drop=False) # Re-numarate the raw indexes because they mixed in the previous step
        yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs) # sum each raw of the actual results with it's previous ones and divide each raw to Total number of actual 1's in the test data
        yy = np.append([0], yy[0:num_count-1]) # add the first curve point (0,0) : for xx=0 we have yy=0
 
        # CALCULATE VALUE CORRESPONDS TO 0.5 (50%)(it's one way to calculate the success of the model)
        percent = 0.5  
        row_index = int(np.trunc(num_count * percent))
        val_y1 = yy[row_index]
        val_y2 = yy[row_index+1]
        if val_y1 == val_y2:
            val = val_y1*1.0 
        else:
            val_x1 = xx[row_index]
            val_x2 = xx[row_index+1]
            val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)
        
        # CALCULATE THE RATIO OF THE AREAS OF THE BEST MODEL AND INPUT MODEL (anothe way to calculate the success of the model)
        sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1 # Area under best possible model
        sigma_model = integrate.simps(yy,xx) # Area under worst possible model
        sigma_random = integrate.simps(xx,xx) # Area under inputmodel
        ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random) # calculate ratio
 
        # return the analyse
        return ideal,xx,yy,val,ar_value