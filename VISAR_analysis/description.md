This folder contains python supports the data analysis done on [Neutrino tool](https://github.com/NeutrinoToolkit/Neutrino) for
Velocity interferometer system for any reflector (VISAR), Streak Optical Pyrometry (SOP) and GOI optical system.

- **export_txt.py (for MACOS users)**

It opens the neutrino session (.neus files), saves the txt file on a specific folder.

 <ins> *Parameters to change* </ins>

```
1. shots_path = "..." # path of the data 
2. dst_folder = "..." # folder destination path
3. path_of_txt = "..." # path for txt files
4. list_of_shots = sorted([s for s in os.listdir(dst_folder) if s.startswith(' ') ],key=lambda x: int(int(re.search(r' ', x).group(1)))) # change accordingly 
```
- **extract_shock_velocities.py (for MACOS users)**
Based on the txt files from [Neutrino](https://github.com/NeutrinoToolkit/Neutrino) analysis, it derives the shock velocities by fitting first order polynomial to a specific range that the user decides.

<ins> *Parameters to change* </ins>
```
1. path_of_txt = "..."
2. path_xlsx = "..."
```
<ins> *Attention* </ins>

The specific class contains two different ways of choosing a preferable region for the linear fit:

* manual_slicing: Need to change the parameters (see below) manually
* interactive_slicing: Need to activate (uncomment) matplotlib.use('qtagg')

When you find the set of parameters, write them on the excel file ( <ins>extract_shock_velocities_slice</ins> )

| Parameters    | Effect        |
| ------------- |:-------------:|
| refni / samni | number of cell|
| parminl       | left border on the reference material slicing| 
| parmaxl       | right border on the reference material slicing|
| parminr       | left border on the sample material slicing| 
| parmaxr       | right border on the sample material slicing|




