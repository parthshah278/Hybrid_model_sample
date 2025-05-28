# Hybrid_model_sample
This code sample shows the training of a hybrid model for a reaction-diffusion process.
The Hybrid model code for PDEs with spatiotemporal parameters The architecture of the hybrid model is attached as follows: Run Hybrid_Model.py file Requires Tensorflow, Pandas, Matplotlib Outputs: The outputs are videos of the spatiotemporal evolution of parameters and output, Outputs: Plots for temporal evolutions of temporal variables, .csv file names K_lambda.csv output to get the values of temporally varying and spatially lumped parameters

The structure of the hybrid model is shown below:
![image](https://github.com/user-attachments/assets/f328ec25-cda6-4519-a797-177ad39e870a)

The architecture for the hybrid model for the reaction-diffusion system with the two branches is attached as follows:
![image](https://github.com/user-attachments/assets/8d198603-bf9a-45ec-b4fc-a7c9da1931d1)

Initial weights for model training are taken from Branch_1.h5 and Branch_2.h5.
