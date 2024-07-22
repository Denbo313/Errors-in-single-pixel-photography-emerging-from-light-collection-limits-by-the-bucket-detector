# Errors-in-single-pixel-photography-emerging-from-light-collection-limits-by-the-bucket-detector
Supplementary data and code supporting the article 'Errors in single pixel photography emerging from light collection limits by the bucket detector'

Dependencies: 
- spgl1, required if CS is desired. https://github.com/mpf/spgl1
- pillow, https://pypi.org/project/pillow/

The following files and folders are in this directory:

- 'cifar-10_batches-py': Contains the cifar-10 data batch that are used for the simulation as a compressed .rar file. **Please extract for usage**
- 'experimental_data': Contains the experimentally measured data as .mat files (readable for matlab and python), divided into 'complex_holograms' (figure 2) and 'tilted lens' (figure 3). **Please extract 'complex_hologram.rar' and 'tilted_lens.rar' for usage** 
- 'results': Contains the images of the reconstructed amplitude, divided into 'experiment' and 'simulation'. **Please run 'simulation_code.py' or extract the rar files in the directories to get the results**
- '_helper_function.py': Contains the functions used in the simulations and evaluation of experimental data.
- 'simulation_code.py': Runs the simulation of the experiment and saves the results in **results/simulation/amp**
- 'reconstruct_experiment.py': Reads and evaluates the experimental data of **experimental_data**. **VERY IMPORTANT: Before usage export the .rar files**. Results are stored with proper naming in **results/experiment**. 

For any questions, please contact: dennis.scheidt@correo.nucleares.unam.mx
