This folder contains python supports the data analysis done on [Neutrino tool](https://github.com/NeutrinoToolkit/Neutrino) for
Velocity interferometer system for any reflector (VISAR), Streak Optical Pyrometry (SOP) and GOI optical system.

- export_txt.py (for MACOS users)

It opens the neutrino session (.neus files), saves the txt file on a specific folder.

**Parameters to change**
'''
1. shots_path = "..." # path of the data 
2. dst_folder = "..." # folder destination path
3. path_of_txt = "..." # path for txt files
4. list_of_shots = sorted([s for s in os.listdir(dst_folder) if s.startswith(' ') ],key=lambda x: int(int(re.search(r' ', x).group(1)))) # change accordingly 
'''
