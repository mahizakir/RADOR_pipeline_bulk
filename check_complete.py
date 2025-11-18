import os

if __name__ == "__main__":
    not_finished = []
    finished = []
    for disease_name in os.listdir('data'):
        file_path = os.path.join('data', disease_name, f'generated_simple_sentences_{disease_name}.csv')
        if os.path.isfile(file_path):
            finished.append(disease_name)
        else:
            not_finished.append(disease_name)
    
    with open('need_compute_disease_folders_names.csv', 'w') as f:
        for name in not_finished:
            f.write(f'{name}\n')

    print(f'Number Finished diseases: {len(finished)}')
    print(f'Number of diseases needing processing: {len(not_finished)}')