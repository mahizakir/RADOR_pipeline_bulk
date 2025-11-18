import os
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--data_dir", type=str, default="data",
                            help="Directory containing disease data folders")
    argparser.add_argument("-o", "--output_dir", type=str, default="collected_results",
                            help="Directory to store output results")
    argparser.add_argument("-f", "--finished_file", type=str, default="finished_disease.csv",
                            help="File to record finished diseases")
    argparser.add_argument("-n", "--need_compute_file", type=str, default="need_compute_disease.csv",
                            help="File to record diseases needing computation")
    argparser.add_argument("-c", "--collect", action='store_true',
                            help="Enable data collection to output_dir")
    
    args = argparser.parse_args()
    print('Collecting results and identifying unprocessed diseases...')

    # scan data directory to find which diseases need processing 
    need_compute_fp = args.need_compute_file
    finished_fp = args.finished_file
    need_scan = True
    
    if os.path.exists(need_compute_fp) and os.path.exists(finished_fp):
        with open(need_compute_fp, 'r') as f:
            need_compute = [line.strip() for line in f.readlines()]
        print(f'Number of diseases needing processing (from file): {len(need_compute)}')
        
        with open(finished_fp, 'r') as f:
            finished = [line.strip() for line in f.readlines()]
        print(f'Number of diseases already finished (from file): {len(finished)}')
    else:
        need_compute = []
        finished = []
        for disease_name in os.listdir('data'):
            file_path = os.path.join('data', disease_name, f'generated_simple_sentences_{disease_name}.csv')
            if os.path.isfile(file_path):
                finished.append(disease_name)
            else:
                need_compute.append(disease_name)
        print(f'Number of diseases needing processing (scanned): {len(need_compute)}')
        print(f'Number of diseases finished (scanned): {len(finished)}')
        need_scan = False
        os.remove(need_compute_fp) if os.path.exists(need_compute_fp) else None
    
    # Append newly finished diseases
    if need_scan:
        print('Checking for newly finished diseases...')
        for disease_name in need_compute:
            file_path = os.path.join('data', disease_name, f'generated_simple_sentences_{disease_name}.csv')
            if os.path.isfile(file_path) and disease_name not in finished:
                finished.append(disease_name)
                if disease_name in need_compute:
                    need_compute.remove(disease_name)
    else:
        print('no need to rescan for finished diseases.')
        
    # save the lists back to files
    finished = list(set(finished))
    need_compute = list(set(need_compute))
    for data_list, file_path in [(finished, finished_fp), (need_compute, need_compute_fp)]:
        print(f'Saving to {file_path}...')
        with open(file_path, 'w') as f:
            for name in data_list:
                f.write(f'{name}\n')

    print(f'Number Finished diseases: {len(finished)}')
    print(f'Number of diseases needing processing: {len(need_compute)}')
    
    if args.collect:
        print('Collecting data to output directory...')
        os.makedirs(args.output_dir, exist_ok=True)
        for disease_name in finished:
            src_file = os.path.join(args.data_dir, disease_name, f'generated_simple_sentences_{disease_name}.csv')
            if os.path.isfile(src_file):
                dest_file = os.path.join(args.output_dir, f'generated_simple_sentences_{disease_name}.csv')
                with open(src_file, 'r') as sf, open(dest_file, 'w') as df:
                    df.write(sf.read())
        print('Data collection completed.')