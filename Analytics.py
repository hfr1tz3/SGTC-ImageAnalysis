import numpy as np
import pandas as pd
import cv2 as cv
import skimage
import matplotlib.pyplot as plt

''' The following is a file to perform qualitative analyses and quickly convert image data into dataframes and whatever other functions I create for this project '''


' Pipeline of all computations to be done on raw track data from `btrack`. '
''' Input: Raw tracks dataframe (type pd.DataFrame)
    Parameters: counts (bool) [default: False]. Produce a count dataframe from
    the `sample_counts` method.
    Output: tracks dataframe with roundess, velocity, speed calculations
'''
def tracks_pipeline(raw_tracks_data, title = None):
    tracks = raw_tracks_data.copy()
    tracks.rename(columns = {'Unnamed: 0': 'index', '0': 'cell', '1': 'frame', '2':'xcoord', '3':'ycoord'}, inplace=True)
    tracks = get_velocities(tracks)
    tracks = get_morphology(tracks)

    frame_stats_df = frame_feature_counts(tracks)
    sample_stats_df = sample_counts(frame_stats_df, title = title)
    return tracks, frame_stats_df, sample_stats_df

' Given a tracks dataframe, compute count statistics (identical to the `sample_ch_stats` method) for cells between frames. '
''' Input: tracks dataframe computed from baesian tracker `btrack`.
    Output: pandas.DataFrame containing cell count statistics for each sample including:
    ----------------- Count Metrics -------------------
    Min cell count
    Max cell count
    Average cell count
    Standard deviation between cell counts over all frames
    Average rate of change of cell counts per frame
    ----------------- Netosis Count Metrics ----------
    Netosis count
    Average Netosis count
    Standard deviation between netosis counts over all frames
    Average rate of change of netosis counts per frame
    ----------------- Roundness Metrics ---------------
    Average cell roundness per frame
    Average rate of change of cell roundness per frame
    Standard deviation between cell roundness over all frames
    ----------------- Gini Index Metrics ------------- (Useful??)
    Average Gini index 
    Max Gini Index
    Min Gini Index
    ---------------- Velocity Metrics ----------------
    Average Velocity per frame
    Average rate of change of velocity per frame
    Standard deviation of velocity over all frames
    Average Speed per frame
    Average rate of change of speed per frame
    Standard deviation of speed over all frames
''' 
def sample_counts(df, title = None):
    sample_stats = {}
    if 'frame 0' in df.index:
        frames = df
    else:
        frames = frame_feature_counts(df)
    mname = 'min_'
    Mname = 'max_'
    sname = 'std_'
    aname = 'avg_'
    rname = '_roc'
    for feature in frames.columns.values:
        if '_roc' not in feature:
            sample_stats[mname+feature] = frames[feature].min()
            sample_stats[Mname+feature] = frames[feature].max()
            sample_stats[aname+feature] = frames[feature].mean()
            sample_stats[sname+feature] = frames[feature].std()
        else:
            sample_stats[aname+feature+rname] = frames[feature].mean()
    sample_df = pd.DataFrame.from_dict(sample_stats, orient = 'index', columns = [f'{title}'])    
    return sample_df

' Write a function to analyze counts and ROC of counts for all features from features tracked with `btrack` '
''' Input: tracks
    Output: sample dataframe with overall sample stats.
'''
def frame_feature_counts(tracks):
    features = ['cell', 'area', 'velocity', 'speed', 'roundness']
    # features = [col for col in tracks.columns]
    # features.remove('index')
    frame_stats = {}
    counts = []
    avgs = []
    aname = 'avg_'
    dname = '_roc'
    for feature in features:
        for i in range(int(tracks['frame'].max())+1):
            frame_stats[f'frame {i}'] = {}
            frame = tracks.loc[tracks['frame']==i]
            f_count = frame[feature].shape[0]
            f_avg = np.mean(frame[feature].apply(eval))
            counts.append(f_count)
            avgs.append(f_avg)
            if feature == 'cell':
                frame_stats[f'frame {i}']['count'] = np.asarray([f_count])
                num_rows = 2040//(4*(frame['diameter'].mean()))
                c = count_cells_in_grid(frame, num_rows = num_rows)
                frame_stats[f'frame {i}']['gini_index'] = np.asarray([Gini(c)])
                if i > 0:
                    frame_stats[f'frame {i}']['avg_count_roc'] = np.asarray([counts[i] - counts[i-1]])
                else:
                    frame_stats[f'frame {i}']['avg_count_roc'] = np.asarray([0])
            else:
                frame_stats[f'frame {i}'][aname+feature] = np.asarray([f_avg])
                if i > 0:
                    frame_stats[f'frame {i}'][aname+feature+dname] = np.asarray([avgs[i] - avgs[i-1]])
                else:
                    frame_stats[f'frame {i}'][aname+feature+dname] = np.asarray([0])
    frame_df = pd.DataFrame.from_dict(frame_stats, orient = 'index', columns = frame_stats['frame 0'].keys())    
    return frame_df


' Compile an entire list of sample stats over every channel of data '
''' Input: (list) List of frames (can be list of arrays or list of data frames).
    Parameters:
        name_list: (list) Default = None. List of (str) names for each channel.
    Output: (Data Frame) Data Frame of sample stats for each channel of the sample.

'''

'Compute the Gini Index'
''' Input: (Array like) cell counts for each entry
    Output: (float) Real valued Gini coefficient.
'''

def Gini(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

'Count the cells across a cubulation of our image'
''' Input: (DataFrame) Cell image data frame
    Parameters:
        num_rows: (int) Number of bins per row. This determines number of bins at (num_rows)**2 which uniformly cover the image.
        graph: (bool) Produce a stacked bar chart of cell counts over the bins. This gives a relative location on the frame for each bin.
        histogram: (bool) Orders bin counts and produces a histogram.
    
    Returns: (Numpy array) Cell counts for each bin.
'''
def count_cells_in_grid(celldf, num_rows = 10, frame_size = 2040, histogram = False, graph = False):
    bins = np.zeros((num_rows, num_rows))
    frame = celldf['frame'].values[0]
    gridxbound = frame_size #celldf['centroid X'].max()
    gridybound = frame_size #celldf['centroid Y'].max()
    grid_size = frame_size #np.max(np.asarray(gridxbound, gridybound))
    for i in range(num_rows):
        for j in range(num_rows):
            cell_count = sum(( celldf['xcoord'] > i*(grid_size/num_rows)) & (celldf['xcoord'] < (i+1)*(grid_size/num_rows)) \
                              & (celldf['ycoord'] > j*(grid_size/num_rows)) & (celldf['ycoord'] < (j+1)*(grid_size/num_rows)))
            bins[i,j] = cell_count
            
    if histogram == True:
        order = [[i,bins.flatten()[i]] for i in range(len(bins.flatten()))]
        order.sort(key = lambda x: x[1])
        sort = np.array(order)
        bin_list = []
        for i in range(sort.shape[0]):
            bin_list.append(f'Bin{int(sort[i,0])}')
        fig, ax = plt.subplots(figsize=(10,5))
        plt.bar(bin_list,sort[:,1])
        plt.title(f'Cell Counts at Frame {frame}')
    
    if graph == True:
        bin_list = [f'Col {i}' for i in range(num_rows)]
        legend = [f'Row {i}' for i in range(num_rows)]
        fig, ax = plt.subplots(figsize = (10,10))
        bottom = np.zeros(num_rows)
        for i in range(num_rows):
            if i == 0:
                plt.bar(bin_list, bins[i], label = legend, bottom = bottom)
                bottom += bins[i]
            else:
                plt.bar(bin_list, bins[i], label = legend, bottom = bottom)
                bottom += bins[i]
            plt.title(f'Cell counts at Frame {frame}')
            plt.legend(legend)
            
    return bins.flatten()

'Count the cells across a frame with a padding moving left to right and down to up'
''' Input: Cell image DataFrame
    Parameters: 
        bin_size: (int) Size of the scanning bin we count cells in. Typically we strive for this to be between 10-20% of the whole frame. Default = 200.
        pad: (int) Number of frame in the intersection between adjacent bins. Default = 50 (25% of bin_size).
        frame_size: (int) Size of the used frame. Our data consists of 2040x2400 pixel frames. Default = 2040
        graph: (bool) Produce a stacked bar chart of cell counts over bins.
    Returns: (Numpy array) Cell counts for each bin.
'''

def pad_count_cells_in_grid(celldf, bin_size = 200, pad = 50, frame_size = 2040, graph = False):
    'Compute number of bins'
    assert (pad < bin_size) & (bin_size < frame_size)
    stride = bin_size - pad
    num_bins = (frame_size-bin_size+1)//stride+2
    bins = np.zeros((num_bins, num_bins))
    for i in range(num_bins):
        for j in range(num_bins):
            cell_count = sum(( celldf['xcoord'] > i*stride) & (celldf['xcoord'] < (((i+1)*bin_size)+(i*stride))) \
                              & (celldf['ycoord'] > j*stride) & (celldf['ycoord'] < ((j+1)*bin_size)+(j*stride)))
            bins[i,j] = cell_count
    
    # The stacked bar chart
    if graph == True:
        frame = celldf['frame'].values[0]
        bin_list = [f'Col {i}' for i in range(num_bins)]
        legend = [f'Row {i}' for i in range(num_bins)]
        fig, ax = plt.subplots(figsize=(10,10))
        bottom = np.zeros(num_bins)
        for i in range(num_bins):
            if i == 0:
                plt.bar(bin_list,bins[i], bottom = bottom)
                bottom += bins[i]
            else:
                plt.bar(bin_list, bins[i], bottom = bottom)
                bottom += bins[i]
        plt.title(f'Cell Counts at Frame {frame}')
        plt.legend(legend)
            
    return bins.flatten()

' Add cell velocities and speeds to the tracking data frame for a sample '
def get_velocities(tracks, histogram = False):
    velocities = []
    speeds = []
    for i in range(int(tracks['cell'].min()),int(tracks['cell'].max())+1):
        cell = tracks.loc[tracks['cell'] == i]
        if cell.shape[0] != 0:
            xvel = np.diff(cell['xcoord'],
                           prepend = cell.at[cell['index'].min(), 'xcoord']
                    )
            yvel = np.diff(cell['ycoord'], 
                           prepend = cell.at[cell['index'].min(), 'ycoord']
                    )
            velocities.extend([(xvel[k],yvel[k]) for k in range(len(xvel))])
        
            for k in range(len(xvel)):      
                speed = np.linalg.norm(np.asarray([xvel[k],yvel[k]]), axis = 0)
                speeds.append(speed)
    tracks['velocity'] = velocities
    tracks['speed'] = np.asarray(speeds)
    
    if histogram == True:
        fig, ax = plt.subplots(figsize=(5,8))
        ax.hist(tracks['speed'], log = True)
        ax.title('Cell speed between frames')
        fig.show()
        
    return tracks

' Compute the average speed of all tracked cells in a `tracks` data frame '
def average_speed(tracks, histogram = False):
    avg_speeds = []
    for i in range(int(tracks['cell'].min()), int(tracks['cell'].max()+1)):
        cell = tracks.loc[tracks['cell'] == i]['speed']
        avg_speeds.append(np.average(cell.values))
    
    if histogram == True:
        fig, ax = plt.subplots(figsize=(5,8))
        ax.hist(avg_speeds, log = True)
        ax.title('Average Cell Speeds')
        
    return np.asarray(avg_speeds)

' Compute the diameter and roundness of each cell '
' Here roundness is computed as 4*pi*A/P^2 '
' Diameter is computed as 0.5*(major_axis_length+minor_axis_length) '
def get_morphology(tracks):
    roundness = []
    diameter = []
    for index, row in tracks.iterrows():
        cell_round = (4*np.pi*row['area'])/(row['perimeter']**2)
        cell_diam = 0.5*(row['major_axis_length']+row['minor_axis_length'])
        roundess.append(cell_round)
        diameter.append(cell_diam)
        
    tracks['roundness'] = np.asarray(roundness)
    tracks['diameter'] = np.asarray(diameter)
    return tracks

''' Using the tracking dataframes, and given a cell in a sample, compute a 4*cell_diameter neighborhood of the given cell in each frame of the tracked sample.

Input: 
`cell` (pd.DataFrame); portion of tracks data frame containing the information for the given cell.
`tracks` (pd.DataFrame); DataFrame of the entire tracked sample.

Output: (pd.DataFrame) containing all cells within 4*cell_diameter over the frames where the given cell is tracked. 
'''
def check_nbhd(cell, tracks, plots = False, max_plots = 0):
    neighborhood = pd.DataFrame()
    for index, row in cell.iterrows():
        frame = row['frame']
        t = tracks.loc[tracks['frame']==frame]
        cell_x, cell_y = row['xcoord'], row['ycoord']
        if 'diameter' not in row.index:
            cell_diameter = 20
        else:
            cell_diameter = row['diameter']
        nbhd_radius = 4*cell_diameter
        nbhd = t.loc[(cell_x - nbhd_radius < t['xcoord']) &
                     (cell_x + nbhd_radius > t['xcoord']) &
                     (cell_y - nbhd_radius < t['ycoord']) &
                     (cell_y + nbhd_radius > t['ycoord'])]
        neighborhood = pd.concat([neighborhood, nbhd], ignore_index = True)
    if plots == True:
        plot_quivers(cell, neighborhood, max_plots = max_plots)
        
    return neighborhood

def plot_quivers(center_cell, neighborhood, max_plots = 20):
    for m,i in enumerate(range(int(neighborhood['frame'].min()), int(neighborhood['frame'].max())+1)):
        frame = neighborhood.loc[neighborhood['frame'] == i]
        center_x = center_cell.loc[center_cell['frame']==i]['xcoord']
        center_y = center_cell.loc[center_cell['frame']==i]['ycoord']
        if 'diameter' not in frame.columns:
            cell_diameter = 20
        else:
            cell_diameter = center_cell.loc[center_cell['frame']==i]['diameter']
        nbhd_radius = 4*cell_diameter
        x,y = frame['xcoord'], frame['ycoord']
        u = [k[0] for k in frame['velocity'].apply(eval).values]
        v = [k[0] for k in frame['velocity'].apply(eval).values]
        fig,ax = plt.subplots()
        ax.quiver(x,y,u,v)
        ax.scatter(x,y,marker='o', color='black')
        ax.plot(center_x,center_y,color='r',marker='o')
        ax.set_xlim(left = np.max([float(center_x) - 1.2*nbhd_radius,0]), right = np.min([float(center_x) + 1.2*nbhd_radius, 2040]))
        ax.set_ylim(bottom = np.max([float(center_y) - 1.2*nbhd_radius,0]), top = np.min([float(center_y) + 1.2*nbhd_radius, 2040]))
        ax.set_title(f'Cell Velocity Frame {i}')
        if m == max_plots-1:
            break
    return fig

# Eventually we should split these into classes

'------------------------------------------------------------------------------------------------'

def cell_intensity(contour):
    M = skimage.measure.moments_coords(contour)
    return M[0,0]

def cell_centroid(contour):
    M = skimage.measure.moments_coords(contour)
    centroid = (M[1,0] / M[0,0], M[0,1] / M[0,0])
    return centroid

def load_frames(Path, num_frames):
    frames = []
    for i in range(num_frames):
        f = f'/frame{i}.npy'
        p = Path + f
        x = np.load(p)
        frames.append(x)
    return frames

' Compute some basic quantitative stats on a set of frames '
'''
For a set of frames (ie. npy files of cell contours), compute basic statistics FOR EACH CELL including
Contour area
Contour perimeter
Contour centroid (x and y coords seperately and as a tuple)

Will add more stats later.

Returns: Embedded list structure
'''

def frame_analysis(frames):
    frame_list = []
    for i in range(len(frames)):
        frame_list.append({})
        frame_list[i]['cell'] = []
        frame_list[i]['area'] = []
        #frame_list[i]['perimeter'] = []
        frame_list[i]['centroid'] = [] 
        frame_list[i]['xcoord'] = []
        frame_list[i]['ycoord'] = []
        for j in range(frames[i].shape[0]):
            frame_list[i]['cell'].append(j)

            a = cv.contourArea(frames[i][j])
            c = cell_centroid(frames[i][j])
            #p = skimage.measure.perimeter(frames[i][j])

            frame_list[i]['area'].append(a)
            #frame_list[i]['perimeter'].append(p)
            frame_list[i]['centroid'].append(c)
            frame_list[i]['xcoord'].append(c[0])
            frame_list[i]['ycoord'].append(c[1])
            
    return frame_list

'Create Pandas DataFrame of the embedded list structure from the frame_analysis method'
def cell_stats(frames):
    frame_list = frame_analysis(frames)
    for i in range(len(frames)):
        if i == 0:
            df = pd.DataFrame(frame_list[i], index=list(np.full((1,len(frame_list[i]['cell'])), i)))
        else:
            dfi = pd.DataFrame(frame_list[i], index=list(np.full((1,len(frame_list[i]['cell'])), i)))
            df = pd.concat([df,dfi])
    df['frame'] = [df.index[k][0] for k in range(df.index.shape[0])]
    df = df[['frame', 'cell', 'area', 'centroid', 'xcoord', 'ycoord']]
            
    return df

' Cell count statistics for a sample (across all frames) '
''' Input: (Data Frame) Cell Contour Data Frame given from cell_stats method.
    Parameters:
        frame_counts: (bool) Default = False. If True, then data frame will contain
        rows for each frame's individual cell count.
    Output: (Data Frame) Data Frame containing cell count statistics for each sample
    including:
    Min cell count
    Max cell count
    Average cell count
    Standard deviation between cell counts over all frames
    Average rate of change of cell counts per frame
    ----------------- Gini Index Metrics ------------- (Useful??)
    Average Gini index 
    Max Gini Index
    Min Gini Index
    (add more?)
    
'''
def sample_ch_stats(frames, title = None, frame_counts = False): # Add graphing parameter options?
    cell_counts = []
    sample_stats = {}
    diffs = []
    ginis = [] ## Would this be useful?
    for i in range(frames['frame'].max()+1):
        diff = 0
        # Compute Gini Index
        frame = frames[frames['frame'] == i]
        c = count_cells_in_grid(frame)
        g = Gini(c)
        ginis.append(g)
        # Count cells
        num_cells = frames.loc[frames['frame'] == i, 'cell']
        cell_counts.append(len(num_cells))
        if frame_counts == True:
            sample_stats[f'frame {i} Count'] = int(len(frame))
        if i == 0:
            continue
        else:
            diff = cell_counts[i]-cell_counts[i-1]
            diffs.append(diff)
    # Type to array
    ginis = np.asarray(ginis)
    diffs = np.asarray(diffs)       
    cell_counts = np.asarray(cell_counts)
    # Compute stats
    sample_stats['Min Count'] = cell_counts.min()
    sample_stats['Max Count'] = cell_counts.max()
    sample_stats['Average Count'] = cell_counts.mean()
    sample_stats['Std Count'] = cell_counts.std()
    sample_stats['Avg Count ROC'] = diffs.mean()
    sample_stats['Min Gini Index'] = ginis.min()
    sample_stats['Max Gini Index'] = ginis.max()
    sample_stats['Avg Gini Index'] = ginis.mean()
    stats = pd.DataFrame.from_dict(data = sample_stats, orient = "index", columns = [f'{title}'])
    
    return stats

def sample_stats(ch_list, name_list = None):
    # We allow the input ch_list to either be a list of data frames
    # They can be cell_stats data frames
    # Or they can be sample_ch_stats data frames
    CH_list = ch_list.copy()
    if type(CH_list[0]) == list: # If list entries are raw numpy data lists
        for i in range(len(CH_list)):
            stats = cell_stats(CH_list[i])
            CH_list[i] = stats # Reassign frame list copy to be a data frame
    if type(CH_list[0]) == pd.core.frame.DataFrame:
        if (CH_list[0]).shape[0] == 8: # sample_ch_stats has 8 rows
            df = pd.concat([CH_list[i] for i in range(len(CH_list))], axis = 1)
        else: # cell_stats data frame does not have 8 rows
            for i in range(len(CH_list)):
                if i == 0:
                    ch1_stats = sample_ch_stats(CH_list[i], title = name_list[i])
                    df = ch1_stats
                else:
                    ch_stats = sample_ch_stats(CH_list[i], title = name_list[i])
                    df = pd.concat([df,ch_stats], axis = 1)
        
    return df 

' Construct a list a Gini coefficents and graph them over an entire sample' 
''' Input: (Data Frame) cell_stats Data Frame
    Parameters:
        num_rows: (int) Default = 10. Number of bins for the count_cells method
        graph: (bool) Default = False. If True, produce a graph of Gini index for a sample.
        sample_name: (string) Default = None. Name the sample for the title of the graph produced. 
    
    Returns: (list) Gini Indexes for every frame in a sample. 
'''
def Gini_list(frames, num_rows = 10, graph = False, sample_name = None):
    ginis = list()
    for i in range(int(frames['frame'].max())+1):
        frame = frames[frames['frame'] == i]
        c = Analytics.count_cells(frame, num_rows = num_rows)
        g = Analytics.Gini(c)
        ginis.append(g)
    if graph == True:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(np.arange(int(frames['frame'].max())+1), ginis)
        plt.title(f'Gini Coefficient across {sample_name} Frames')
        fig.show()
    return np.asarray(ginis)