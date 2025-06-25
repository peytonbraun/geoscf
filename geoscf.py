import os, sys, glob
import requests
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
class GEOSCF:
    def __init__(self, base_url='https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v1/das/', 
                 output_dir_pattern='GEOS_CF/Y%Y/M%m/D%d'):
        '''
        base_url:
            like https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v1/ana/
        output_dir_pattern:
            datetime pattern of dir structure to save data
        '''
        self.logger = logging.getLogger(__name__)
        self.base_url = base_url.rstrip('/') #if it has a trailing '/', remove it
        self.output_dir_pattern = output_dir_pattern
    
    def download_and_resave(self,if_download=True,if_delete=True,
                            generate_filenames_kw=None,
                            download_file_kw=None,
                            west=-170.,east=-30,south=10.,north=85.,
                            lev_slice_collection=[None, None],
                            variables_collection=[
                                ['TROPCOL_CO', 'AOD550_BC', 'AOD550_OC', 'AOD550_DUST'],
                                ['EMIS_BCPI_BB', 'EMIS_BCPP_BB', 'EMIS_OCPU_BB', 'EMIS_OCPO_BB']
                            ],
                            max_workers=2
                           ):
        '''workhorse, (optionally) download by file, load, subset, and save
        if_download:
            download if true
        if_delete:
            delete raw file if true
        generate_filenames_kw:
            dict, keyword arguments to function generate_filenames
        download_file_kw:
            input to download_file. default is good
        wesn:
            lon/lat boundaries
        lev_slice_collection:
            a list of None or slice. if a list, shall be the same length as collections
            input to generate_filenames
        variables_collection:
            a list of None or list of variable names. same length as collections
        '''
        generate_filenames_kw = generate_filenames_kw or {}
        download_file_kw = download_file_kw or {}
        filenames,timestamps,collections = self.generate_filenames(**generate_filenames_kw)
        
        if lev_slice_collection is None:
            lev_slice_collection = [None for _ in range(len(collections))]
        if variables_collection is None:
            variables_collection = [None for _ in range(len(collections))]

        if if_download:
            all_fns = filenames.flatten().tolist()
            self.download_files_parallel_with_retry(all_fns, max_workers, **download_file_kw)

        for itime,timestamp in enumerate(timestamps):
            for ic,collection in enumerate(collections):
                fn = filenames[itime,ic]
                local_path = os.path.join(timestamp.strftime(self.output_dir_pattern), fn)
            
                ds = self.load_dataset(local_path, west=west, east=east, south=south, north=north,
                                       lev_slice=lev_slice_collection[ic],
                                       variables=variables_collection[ic])
                if ds is None:
                    self.logger.warning(f"Skipping file due to failed load: {local_path}")
                    continue
                save_fn = fn.replace('g1440x721',
                                     'g{}x{}'.format(ds.sizes['lon'],ds.sizes['lat']))
                # not very decent solution
                if 'v36' in save_fn:
                    save_fn = save_fn.replace('v36','v{}'.format(ds.sizes['lev']))
                save_path = os.path.join(timestamp.strftime(self.output_dir_pattern),save_fn)
                ds.to_netcdf(save_path,format='NETCDF4')
                #self.logger.info(f"Saving dataset to: {save_path}")
                ds.close() #changed
                if if_delete:
                    os.remove(local_path)
    
    def download_file(self,fn,chunk_size=None,stream=False):
        '''download a single file
        keep chunk_size and stream off unless files are too big for your memory
        '''
        #use the base url from earlier and add the filename details
        timestamp = dt.datetime.strptime(fn[-18:-5],"%Y%m%d_%H%M")
        output_dir = timestamp.strftime(self.output_dir_pattern)
        
        url = "{}/{}/{}".format(self.base_url,timestamp.strftime('Y%Y/M%m/D%d'),fn) 
        #If there isn't a downloads folder, make one
        os.makedirs(output_dir, exist_ok=True) 
        local_path = os.path.join(output_dir, fn)
        self.logger.info(f"Downloading: {url}") 

        #stream download so there's no memory issues
        response = requests.get(url, stream=stream) 
        #handle what happens if the file isn't there
        if response.status_code == 404: 
            self.logger.warning(f"File not found: {fn}")
            return local_path
        #automatically checks if the request succeeded and if not, says what happened
        response.raise_for_status() 

        #open the file, give it the name 'f'
        with open(local_path, "wb") as f: 
            if chunk_size is None:
                f.write(response.content)
            else:
                #basically processes the file in chunks, which is more efficient for memory
                for chunk in response.iter_content(chunk_size=chunk_size): 
                    f.write(chunk) #building up the file piece by piece
        #self.logger.info(f"Saved to: {local_path}")
        return local_path
    
    def download_files(self,fns,**kwargs):
        '''download multiple files.'''
        local_paths = []
        for fname in fns:
            try: #if one file fails, continues to download the rest of the files
                local_path = self.download_file(fname, **kwargs)
                if local_path:
                    local_paths.append(local_path)
            except Exception as e:
                self.logger.warning(f"Failed to download {fname}: {e}")
        return local_paths
    
    def load_dataset(self,local_path,west=-170.,east=-30,south=10.,north=85.,lev_slice=None,
                     variables=None
                    ):
        '''load and subset data from a single file
        wesn:
            lon/lat boundaries
        lev_slice:
            a slice, reserved to subset vertically
        variables:
            variables to load
        '''
        #self.logger.info(f"Loading dataset from: {local_path}")
        if not os.path.exists(local_path):
            self.logger.warning(f'{local_path} does not exist!')
            return
        ds = xr.open_dataset(local_path)
        
         # Always slice lon/lat by boundaries
        indexers = dict(lon=slice(west, east), lat=slice(south, north))

        # If lev_slice is given as a list of levels, select those levels explicitly
        if lev_slice is not None:
            indexers['lev'] = lev_slice  # select those exact levels
        
        ds_sel = ds.sel(indexers)
        if variables is None:
            return ds_sel.load()
        else:
            variables = [v for v in ds.data_vars if v in variables]
            if not variables:
                self.logger.warning(f"No matching variables in {local_path}, return all")
                return ds_sel.load()
            else:
                return ds_sel[variables].load()

    @staticmethod
    def generate_filenames(fn_template='GEOS-CF.{}.{}.{}.{}.nc4',
                           versions=None,modes=None,
                           collections=[
                               'met_tavg_1hr_g1440x721_x1',
                               'ems_tavg_1hr_g1440x721_x1'
                           ],
                           timestamps=None,
                           start=None,end=None,period=None,freq=None
                          ):
        '''generate a list of GEOS-CF filenames based on collections and time range
        see section 5.1 at https://gmao.gsfc.nasa.gov/pubs/docs/Knowland1446.pdf
        versions,modes:
            by default 'v01' and 'rpl'
        collections:
            lists of geos cf collections
        return:
            filenames as a 2d array, len(timestamps),len(collections)
            timestamps and collections
        '''
        # chm_tavg_1hr_g1440x721_v36: 2.9G
        # met_tavg_1hr_g1440x721_v36: 0.6G
        if versions is None:
            versions = ['v01' for _ in range(len(collections))]
        if modes is None:
            modes = ['rpl' for _ in range(len(collections))]
        #if timestamps is None:
            #timestamps = pd.date_range(start,end,period,freq)
        if timestamps is None:
            if start is not None and (end is not None or period is not None):
                timestamps = pd.date_range(start=start, end=end, periods=period, freq=freq)
            else:
                raise ValueError("Must provide either 'timestamps' or both 'start' and 'end' with 'freq'")
        filenames = np.empty((len(timestamps),len(collections)),dtype=object)
        
        for ic,collection in enumerate(collections):
            for itime,timestamp in enumerate(timestamps):
                tstr = timestamp.strftime("%Y%m%d_%H%Mz")
                fn = fn_template.format(versions[ic],modes[ic],collection,tstr)
                filenames[itime,ic] = fn
        return filenames,timestamps,collections

    def download_file_with_retry(self, fn, retries=5, delay=5, **kwargs):
        '''
        Download a single file with retry logic to handle transient errors like
        connection timeouts or network glitches.
    
        Parameters:
            fn (str): Filename to download.
            retries (int): Number of retry attempts before giving up.
            delay (int or float): Seconds to wait between retries.
            **kwargs: Additional keyword arguments passed to download_file.
    
        Returns:
            local_path (str or None): Path to downloaded file if successful,
                                      None if all retries failed.
        '''
        for i in range(retries):
            try:
                # Try downloading the file normally
                return self.download_file(fn, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as e:
                # If a connection error or timeout occurs, log it and retry after delay
                self.logger.warning(f"Download failed for {fn} on attempt {i+1}/{retries}: {e}")
                time.sleep(delay)
        # If all retries exhausted without success, log error and return None
        self.logger.error(f"All retries failed for {fn}")
        return None

    
    def download_files_parallel_with_retry(self, fns, max_workers, **kwargs):
        '''
        Download multiple files in parallel using threads with retry logic.
    
        Parameters:
            fns (list): List of filenames to download.
            max_workers (int): Maximum number of concurrent threads.
            **kwargs: Additional arguments passed to download_file_with_retry.
    
        Returns:
            local_paths (list): List of paths to successfully downloaded files.
        '''
        local_paths = []  # To store successful download paths
        completed = 0
    
        # Use ThreadPoolExecutor to manage a pool of worker threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks to executor for each filename
            future_to_fn = {executor.submit(self.download_file_with_retry, fn, **kwargs): fn for fn in fns}
            # As each thread completes, process the results
            for future in as_completed(future_to_fn):
                fn = future_to_fn[future]
                try:
                    result = future.result()  # Get the result of download_file_with_retry
                    if result:
                        local_paths.append(result)  # Append successful downloads
                    completed += 1
                    self.logger.info(f"Downloaded {completed}/{len(fns)}: {fn}")  # Log progress
                except Exception as e:
                    # Log if any unexpected exceptions occur during download
                    self.logger.warning(f"Failed to download {fn}: {e}")
        return local_paths
