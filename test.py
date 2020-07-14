""""Same structure as driver.py:
NOTE:
This is not working at all!!!!
Just a rough structure yet"""

# load data

# Load model.
tqdm.write('Loading 12ECG model...')
model = load_12ECG_model()
tqdm.write("done!")

# Iterate over files.
tqdm.write('Extracting 12ECG features...')
num_files = len(input_files)

for i, f in enumerate(input_files):
    print('    {}/{}...'.format(i + 1, num_files))
    tmp_input_file = os.path.join(input_directory, f)
    data, header_data = load_challenge_data(tmp_input_file)
    current_label, current_score = run_12ECG_classifier(data, header_data, classes, model)
    # Save results.

# evaluate