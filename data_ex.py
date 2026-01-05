python -c "
import kagglehub
import shutil
import os

# 1. Download the dataset
path = kagglehub.dataset_download('yasserh/loan-default-dataset')

# 2. Find the CSV and move it to your project folder
destination = 'data/Loan_Default.csv'
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
shutil.move(os.path.join(path, csv_file), destination)

print(f'\nSuccess! Dataset moved to: {destination}')
"
