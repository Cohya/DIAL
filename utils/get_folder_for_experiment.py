
import os
from datetime import datetime
from  pathlib import Path 
from typing import Dict

def get_folder_for_experiment(experiment_config:Dict)-> Path:


    # Create a folder name based on the experiment configuration
    folder_name = Path(f"experiments/{experiment_config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create the directory if it does not exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the configuration file in the folder
    config_file_path = folder_name / 'config.yaml'
    with open(config_file_path, 'w') as file:
        for key, value in experiment_config.items():
            file.write(f"{key}: {value}\n")
    print(f"Experiment folder created: {folder_name}")

    return folder_name

