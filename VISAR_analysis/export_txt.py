

import os 
import shutil
import re
import subprocess


#### MACOS script ##############################################################################################################################
def copy_files_create_txt(shots_path,dst_folder):

    shutil.copytree(shots_path,dst_folder)
    list_of_shots = sorted([s for s in os.listdir(dst_folder) if s.startswith(' ') ],key=lambda x: int(int(re.search(r' ', x).group(1))))
    filename_list = []
    number_of_shot = []
    for shot in list_of_shots:
        filename_list.append(dst_folder+'/'+shot+'/'+'...'+str(int(re.search(r'...', shot).group(1)))+'_SB.neus')
        number_of_shot.append(int(re.search(r'...', shot).group(1)))
    for i,f in enumerate(filename_list):
        subprocess.run(["open", f])  
        applescript = f'''

        -- Step 1: Activate the Neutrino application
        tell application "Neutrino" to activate
        delay 5 -- put some delay for neutrino to fully function

        -- Step 2: Run a background macOS application that allows scripts to interact with the system itself
        tell application "System Events"

            tell process "Neutrino"

                -- Step 3: Wait until a window containing VISAR exists
                set visarWindow to missing value
                repeat
                    delay 1 -- give UI time to update
                    repeat with w in windows
                        -- Check if the window actually has a name property
                        if (exists name of w) then
                            if name of w contains "VISAR" then
                                set visarWindow to w
                                exit repeat
                            end if
                        end if
                    end repeat
                    if visarWindow is not missing value then exit repeat
                end repeat


                -- Step 4: Bring that window to front
                perform action "AXRaise" of visarWindow
                delay 0.2

                -- Step 5: Send Command + Shift + T
                keystroke "t" using {{command down, shift down}}

                -- Step 6: Find the window whose role is Save/Open
                set saveDialog to first window whose subrole is "AXDialog"

                -- Step 7: Set file path or the name
                set filePath to "shot_{number_of_shot[i]}.txt"
            
                -- Step 8: Choose desktop for saving the data
                keystroke "d" using {{command down, shift down}}
                delay 0.5

                -- Step 9: Write the name of the file 
                set value of text field 1 of saveDialog to filePath
                delay 0.2

                -- Step 10: Click the Save button
                click button "Save" of saveDialog

                -- Step 11: Close all windows
                keystroke "q" using {{command down}}


            end tell

        end tell
        tell application "Neutrino" to quit

        '''
        subprocess.run(["osascript", "-e", applescript])


    path_of_txt = " "
    txt_files=[f for f in os.listdir(path_of_txt) if f.startswith('shot_')]
    path_to_save = " "
    os.makedirs(path_to_save)
    

    for txt in txt_files:
        shutil.copyfile(os.path.join(path_of_txt,txt),os.path.join(path_to_save,txt))
        os.remove(os.path.join(path_of_txt,txt))

    return path_to_save

def create_txt(flag): #enable the function <<copy_files_create_txt>>
    if flag:
        shots_path = "..." #path of the data 
        dst_folder = "..." #destination path
        path_of_txt = copy_files_create_txt(shots_path,dst_folder)
        return path_of_txt
    else:
        None 