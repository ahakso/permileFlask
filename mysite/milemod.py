import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.style
import matplotlib as mpl
import matplotlib.pyplot as plt



def nearest_neighbors(df, tgt_make, tgt_model, tgt_year, n_neighbors=20):
#     Pass in a target make/model and a full feature vector dataframe get similar models/years
# 
#   Preprocessing happens here, change this function to change variables and such
# Get an input frame with just columns for knn
# 
# X: ultimate input from with just the KNN variables
# neighbors: all columns of df, with just the full rows
# point: target car which others are similar to 
# 
    tgt_year = int(tgt_year)
    tgt_columns = ['weight','msrp','seats']
    X_frame = df.loc[:,tgt_columns]
    full_row = X_frame.isna().sum(axis=1)==0
    neighbors = df.loc[full_row,:].copy()
    neighbors = neighbors.reset_index()

    # Scale the neighbors, keeping in a frame
    x = neighbors.loc[:,tgt_columns].values
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(x)
    X = pd.DataFrame(X,columns=X_frame.columns)
    
#     Reduce the importance of msrp as a value
    X.msrp = X.msrp/2
    
#     Get the point of the target make and model
    point_idx = (neighbors.make == tgt_make) & (neighbors.model == tgt_model) & (neighbors.year==tgt_year)
    point = X.loc[point_idx,tgt_columns]
    tgt_class = neighbors.vclass.loc[point_idx]

#     Get the index of cars with common classes
    common_class_idx = neighbors.vclass == tgt_class.values[0]
    
    X = X.loc[common_class_idx,:]
    neighbs = neighbors.loc[common_class_idx,:]
#     Reduce possible neighbors to those of the same class
    
    # Get distances for target metrics
    distance = X.copy()
    for i in range(point.shape[1]):
        distance.iloc[:,i] = distance.iloc[:,i]-point.iloc[:,i].values[0]
#     set_trace()
    distance = np.asarray(distance)   
    sort_idx = np.argsort(np.sum((distance**2),1))
    neighbs = neighbs.iloc[sort_idx,:]
    cars_to_return = n_neighbors
    models_included = 0
    while models_included < n_neighbors:
        return_neighbs = neighbs.iloc[0:cars_to_return,:]
        models_included = return_neighbs.model.nunique()
#         print(models_included)
        cars_to_return += n_neighbors-models_included
    return_neighbs_all_models = return_neighbs
    return_neighbs = return_neighbs[return_neighbs.total.groupby([return_neighbs.make,return_neighbs.model]).apply(lambda x: x == x.min())]
    return return_neighbs, return_neighbs_all_models

class CustomSeries(pd.Series):
    @property
    def _constructor(self):
        return CustomSeries
 
    def custom_series_function(self):
        return 'OK'

class CustomDataFrame(pd.DataFrame):
    "My custom dataframe"
    def __init__(self, *args, **kw):
        super(CustomDataFrame, self).__init__(*args, **kw)
 
    @property
    def _constructor(self):
        return CustomDataFrame
 
    _constructor_sliced = CustomSeries
 
    def split_name(self,to_split='name'): 
#         This split in place
        to_split_series = self.loc[:,to_split]
        #     Add make and model from name column
        make,model = zip(*to_split_series.str.split(' ',1))
        model = [x.strip() for x in model]
        self.loc[:,'make'] = make
        self.loc[:,'model'] = model
        return self
    
    
    def missing_summary(self):
        n_models = self.shape[0]
        return pd.DataFrame(self.isnull().sum()/n_models).transpose()
    
    
    def match_make_model_column(self,mm,target_column):
#         args: take in a dataframe with makes and models and a target column to match
#         return: series of matched values, dataframe  mapping original to matched names

        
#         Get rid of confounding dashes
        
        string = [re.sub('-','',x) for x in string]
        model = re.sub('-','',model)

    #     Match make anywhere
        make_match = np.asarray([bool(re.search(make,x,re.IGNORECASE)) for x in string])
    #     Match model at end of line or before space    
        model_match = np.asarray([bool(re.search('{}$|{} '.format(model,model),x,re.IGNORECASE)) \
                                  for x in string])

    #     Get the index of matching element
        match_idx = np.nonzero((model_match) & (make_match))[0]
        if len(match_idx) == 0:
    #         print('{} {} Not Found'.format(make,model))
            return np.nan
        else:
    #         print('{} {}\n{}'.format(make,model,string[match_idx[0]]))
            return match_idx
    

    def supplement(self,df_add,column_add,verbose=False):
#         Requires that both old and new dataframes have columns 'make' and 'model'
#         Self is changed in place

        df_add = df_add.copy()
#         Get the type
        col_dtype = type(df_add.loc[:,column_add].iloc[0])
#         Add the column if it doesn't exist yet
        if column_add not in self.columns:
            self.loc[:,column_add] = np.nan            
#         Get the makes and models with unmatched values
        to_match_idx = self.loc[:,column_add].isnull()
        to_match = self.loc[to_match_idx,('make','model')]
        if column_add == 'model':
            to_match.model = to_match.model.str.replace('-','')
            df_add.model = df_add.model.str.replace('-','')
#         Get the index of the match in the df_add frame
        def get_index(f_make, f_mdl):
            match_model_idx = np.asarray([bool(re.search(' {}$| {}|^{}'.format(f_mdl,f_mdl,f_mdl),x)) for x in df_add.model])
            match_make_idx = np.asarray([bool(re.search('^{}'.format(f_make),x)) for x in df_add.make])
            match_idx = match_model_idx & match_make_idx
            match_scalar = np.nonzero(np.asarray(match_idx))[0]              
            return match_scalar
        def convert_indices(x):
            if len(x) == 0:
                x = np.nan
            return x
        def index_med_val(idx_array,df_,column):
            
            if np.isnan(idx_array).all() or df_.loc[:,column].iloc[idx_array].isnull().all(): #no match or all nans
                return np.nan

        #     Takes an array of indices into input dataframe and outputs median value
            test_idx = 0
            while_val = True

            while while_val:
                try:
                    first_valid_val = df_.loc[:,column].iloc[idx_array[test_idx]]
                    if isinstance(first_valid_val,str):
                        while_val = False
                    else:
                        while_val = np.isnan(first_valid_val)
                except:
                    set_trace()
                test_idx += 1
            if isinstance(first_valid_val,float):
                try:        
                    return np.nanmedian(df_.loc[:,column].iloc[idx_array].astype('float'))
                except ValueError:
                    set_trace()
            elif isinstance(first_valid_val,str):
                try:                    
                    return first_valid_val
                except:
                    set_trace()
                
#         Get the indices of matched models
        match_idx = [get_index(f_make, f_mdl) for f_make,f_mdl in zip(to_match.make,to_match.model)]
        match_idx = [convert_indices(x) for x in match_idx]  
#         Get the median values, ignoring nans        
        med_values = np.asarray([index_med_val(x,df_add,column_add) for x in match_idx])

    
#         Assign the median values
        df.loc[to_match_idx,column_add] = med_values
        mdlmap = dict(zip(to_match.model, match_idx)) #to_match.make + ' ' +
        if verbose:
            print('Filled {}% of values'.format(100-100*sum(np.isnan(med_values))/len(to_match)))
        if column_add == 'vclass':
            df.loc[df.vclass=='nan',column_add] = np.nan
        return match_idx, to_match, to_match_idx, mdlmap
#         print(df_add.loc[:,['make','model']].iloc[match_idx[0]])


def context_hist(neighbs, neighbs_all, tgt_make, tgt_model, tgt_year):
    # Make histogram
    mpl.style.use('seaborn')
    costs = neighbs.total.values
    fig = plt.figure()
    fig.patch.set_alpha(0.5)
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.5)
    yl = ax.get_ylim()
    count,bins,_ = ax.hist(costs,edgecolor='black')
    ax.set_xlabel('$/mile',fontsize=20)
    plt.tick_params(labelsize=18)

    # Label bars
    idx = (neighbs_all.make==tgt_make) & (neighbs_all.model==tgt_model) & (neighbs_all.year==tgt_year)
    cost_tgt = neighbs_all.loc[idx,'total'].values[0]
    idx_min = np.argmin(neighbs.total.values)
    idx_max = np.argmax(neighbs.total.values)
    (cost_tgt >= bins) & (cost_tgt <=bins)
    sample_costs = (cost_tgt,costs[idx_min],costs[idx_max])
    mdlstrs = ('Your {}'.format(tgt_model),\
               '{} {}\n{}'.format(neighbs.year.iloc[idx_min],neighbs.make.iloc[idx_min],neighbs.model.iloc[idx_min]),\
               '{} {}\n{}'.format(neighbs.year.iloc[idx_max],neighbs.make.iloc[idx_max],neighbs.model.iloc[idx_max]))
    for (cost, mdlstr) in zip(sample_costs, mdlstrs):
        bar_idx = int(np.nonzero((cost >= bins[:-1]) & (cost <= bins[1:]))[0][0])
        xx = np.mean([bins[bar_idx],bins[bar_idx+1]])
    #     xx = bins[bar_idx]
        yy = count[bar_idx]+yl[1]/10
        plt.text(xx,yy,mdlstr,fontsize=18,ha='center')
    return ax        
        
