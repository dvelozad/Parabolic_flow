import subprocess
import os
import pandas as pd
import numpy as np

def get_excess_chemical_potential_slab(spartial_output_folder, HY_size, HY_zone_inf, n_last_points = None):
    ## Load data
    file_density = f'{spartial_output_folder}/Mean_Comp_Density_Appended.txt'
    file_energy  = f'{spartial_output_folder}/Mean_Comp_Energy_AT.txt'

    df_density_data = pd.read_csv(file_density, sep='\t', names=['nbin', 'x', 'rho'])
    df_energy_data = pd.read_csv(file_energy, sep='\t', names=['ntime', 'energy'])

    ## Setup HY zone
    HY_zone_sup = HY_zone_inf + HY_size

    ## Get density integral
    mask = (df_density_data.x > HY_zone_inf) & (df_density_data.x < HY_zone_sup)
    df_density_HY = df_density_data[mask].copy()

    df_density_HY['integral_index'] = df_density_HY.groupby(by=['nbin', 'x']).cumcount()

    density_integral = (df_density_HY.groupby(by=['integral_index'])[['integral_index', 'x', 'rho']]
                                     .apply(lambda group: np.trapz(group['rho'], x=group.x))
                                     .reset_index(name='integral_value'))
  
    ## Get lambda function gradient
    nu = 7
    Deltal = 1 / np.shape(df_energy_data)[0]
    curveParameter = np.arange(0, 1, Deltal)
    gradLambda = nu * curveParameter **( nu - 1 ) ## Power law Lambda ** 7
    dForce  = Deltal * (gradLambda * df_energy_data.energy).sum()

    ## Use last n_last_points points to calculate mu_A and s_A
    if n_last_points == None:
        n_last_points = int(np.shape(density_integral)[0] * 0.25)

    ## Get mu_A, s_A
    mu_A = (density_integral.iloc[-n_last_points:].integral_value + dForce).mean()
    s_A  = (density_integral.iloc[-n_last_points:].integral_value + dForce).std()

    ## save data in a dict
    excess_chemical_info = {'density_integral'  : density_integral, 
                            'dForce'            : dForce, 
                            'mu_A'              : mu_A, 
                            's_A'               : s_A}

    return excess_chemical_info

def get_denstiy_profile(file_density_profile):
    with open(file_density_profile, 'r') as file:
        lines = file.readlines()[3:]

    segments = []
    current_segment = []
    timestep = None

    # Process each line
    for line in lines:
        ## detect initial row and its not data point
        if line.strip() and not line.startswith(' '):
            if current_segment:
                segments.append((timestep, pd.DataFrame(current_segment, columns=['Bin', 'Coord', 'Ncount', 'density/number'])))
                current_segment = []
            timestep = int(line.split()[0])
        else:
            # Add line to current segment, if it's not a comment or empty line
            if line.strip() and not line.startswith('#'):
                current_segment.append(list(map(float, line.split())))

    if current_segment:
        segments.append((timestep, pd.DataFrame(current_segment, columns=['Bin', 'Coord', 'Ncount', 'density/number'])))

    dataframes_dict = {timestep: df for timestep, df in segments}

    return dataframes_dict


def write_submit_jobs_SLURM(experiment_label, file_path):
    # Generate the SLURM submission script as a string
    script = f"""#!/bin/bash
#SBATCH -o {experiment_label}.out
#SBATCH -e {experiment_label}.err
#SBATCH -J {experiment_label}

#SBATCH --partition=CPU_Std32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-core=1

#SBATCH --mail-type=END
#SBATCH --mail-user=diazd@mpip-mainz.mpg.de

# Wall clock limit:
#SBATCH --time=32:00:00

mpirun -np 32 lmp_mpi -in {experiment_label}.in
"""
    with open(file_path, 'w') as file:
        file.write(script)

    #print(f'Submit file for {experiment_label} written')
        
def submit_jobs_SLURM(folder, submit_file):
    cmd = "".join(
        [
            "cd ",
            folder,
            " && ",
            "sbatch ",
            submit_file
        ]
    )
    subprocess.call(cmd, shell = True)

def create_experiment_folder(output_folder):
    for directory in [output_folder]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Directory '%s' created" %directory)