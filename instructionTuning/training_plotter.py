import pandas as pd
import matplotlib.pyplot as plt

class TrainingPlotter:
    """
    A class designed to load granular training loss data (Step, Loss) 
    and generate a continuous plot across multiple epochs.
    """
    def __init__(self, csv_path):
        """
        Initializes the plotter by loading the data from the specified CSV path.
        Assumes the CSV contains 'Epoch', 'Step', and 'Loss' columns.
        """
        try:
            self.df = pd.read_csv(csv_path)
            if self.df.empty:
                raise ValueError("The loaded CSV file is empty.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: CSV file not found at '{csv_path}'.")
        except ValueError as e:
            raise ValueError(f"Error processing CSV: {e}")

    def plot_step_loss(self, save_filename='training_loss_plot.png'):
        """
        Generates and saves the continuous training loss plot.
        """
        plt.figure(figsize=(12, 6))
        
        # Determine the total offset for subsequent epochs
        df_epoch1 = self.df[self.df['Epoch'] == 1].copy()
        
        # Calculate the starting point for Epoch 2 by adding the last step + the step interval (50)
        epoch1_max_step = df_epoch1['Step'].max()
        if len(df_epoch1) > 1:
            epoch1_step_interval = df_epoch1['Step'].iloc[1] - df_epoch1['Step'].iloc[0]
        else:
            epoch1_step_interval = 0
            
        offset = epoch1_max_step + epoch1_step_interval

        # Plot Epoch 1
        plt.plot(df_epoch1['Step'], df_epoch1['Loss'], 
                 label=f'Epoch 1 Training Loss', 
                 linestyle='-', alpha=0.7)
        
        # Plot subsequent epochs
        for epoch in self.df['Epoch'].unique():
            if epoch == 1:
                continue
            
            df_epoch = self.df[self.df['Epoch'] == epoch].copy()
            
            # Create a continuous X-axis column
            df_epoch['Total_Step'] = df_epoch['Step'] + offset
            
            # Plot the continuous data
            plt.plot(df_epoch['Total_Step'], df_epoch['Loss'], 
                     label=f'Epoch {epoch} Training Loss', 
                     linestyle='-', alpha=0.7)
            
            # Update offset for the next epoch
            offset = df_epoch['Total_Step'].max() + epoch1_step_interval


        plt.title('Training Loss', fontsize=16)
        plt.xlabel('Total Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_filename)
        plt.show() # Display the plot
        
        return save_filename

if __name__ == "__main__":
    
    csv_file = 'instructionTuning/training_loss.csv'
    
    try:
        plotter = TrainingPlotter(csv_file)
        plot_filename = plotter.plot_step_loss()
        print(f"\nPlot saved successfully as: {plot_filename}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"\nCould not run plotter: {e}. Please ensure the CSV file is saved correctly.")