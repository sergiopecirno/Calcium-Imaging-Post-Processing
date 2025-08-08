import os 
from termcolor import colored
import re

def get_base():
    "Returns base path to computer"
    base = os.path.expanduser("~")
    return base 

def find_folder_os(start_path,folder):
    """
    Searches for a folder with a specific name 
    start_path - root search 
    Folder -  name of folder 
    """
    found = [] 
    for root, dirs, files in os.walk(start_path):
        if folder in dirs: 
            found.append(os.path.join(root,folder))
    if not found:
        raise FileNotFoundError(colored(f"{folder}, Does not exist. \n","cyan"))
    return found

def get_folder_files(path_to):
    """
    This function returns two outputs: 
    *** fp - the full path to a directory's contents 
    *** file_end - the name of the directory itsef 
    """
    # os.path.abspath(relative_path)
    content = os.listdir(os.path.abspath(path_to))
    file_end = [f for f in content if os.path.isdir(os.path.join(path_to,f))] 
    fp = [os.path.join(path_to,f) for f in file_end] 
    return fp,file_end

def get_file_paths(path_to,type,sort='none'):
    """
    This function Returns two outputs  
    *** fp - the full path to the file  
    *** file_end - the name of the file itself (xxx.type)

    *** Type - returns the type of files 
        *** Uses get folder files if type is folder 
    *** Sort - sorts by either 'none', 'mtime-modification order','ctime-createion order'
    """
    content = os.listdir(path_to)
    if type == "folder":
        fp, file_end = get_folder_files(path_to)
    else: 
        content = os.listdir(path_to)
        fp = [os.path.join(path_to,f) for f in content if f.endswith(type)]
        file_end = [os.path.basename(f) for f in fp]

    if not fp:
        raise FileNotFoundError(colored(f"{type}, No Files with this exntension exist. \n","cyan"))
    
    if sort == 'name':
        zipped = sorted(zip(fp, file_end), key=lambda x: x[1])
    elif sort == 'number':
        try:
            zipped = sorted(zip(fp, file_end), key=lambda x: float(re.findall(r"\d+", x[1])[0]))
        except IndexError:
            raise ValueError(colored("Filename does not contain a number for sorting. \n", "red"))
    elif sort == 'mtime':
        zipped = sorted(zip(fp, file_end), key=lambda x: os.path.getmtime(x[0]))
    elif sort == 'ctime':
        zipped = sorted(zip(fp, file_end), key=lambda x: os.path.getctime(x[0]))
    elif sort == 'none':
        zipped = list(zip(fp, file_end))
    else:
        raise ValueError(colored(f"Invalid sort type: {sort}. Use 'none',name,number 'mtime', or 'ctime'. \n", "red"))

    # Unzip and return as two lists
    fp_sorted, file_end_sorted = zip(*zipped)
    return list(fp_sorted), list(file_end_sorted)

def mk_dir(dir_name,location="current"):
    """
    Function creates new directory in current wd
    """
    if location =="current":
        location = os.getcwd()
    else:
        if not os.path.exists(location):
            raise FileNotFoundError(colored(f"Directory not found: {location}","cyan"))
    dir_path = os.path.join(location,dir_name)
    try:
        os.mkdir(dir_path)
        print(colored(f"Directory '{os.path.abspath(dir_name)}' created successfully. \n","cyan"))
        return dir_path
    except FileExistsError:
        print(colored(f"Directory '{os.path.abspath(dir_name)}' already exists. \n","cyan"))
        return dir_path
    except PermissionError:
        print(colored(f"Permission denied: Unable to create '{os.path.abspath(dir_name)}'. \n","cyan"))
    except Exception as e:
        print(colored(f"An error occurred: {e} \n","cyan"))

    return dir_path



