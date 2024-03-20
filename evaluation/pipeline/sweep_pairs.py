import argparse, os, subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dominance_prediction_dir', default=None, type=str)
    parser.add_argument('--hi_prediction_dir', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--ground_truth_hi_path', default=None, type=str)
    parser.add_argument('--ground_truth_prob_path', default=None, type=str)
    parser.add_argument('--force', action="store_true")
    parser.add_argument('--sequence_file', default=None, type=str)
    args = parser.parse_args()
    return args

args = parse_args()

dominance_prediction_files = []
for root, dirs, files in os.walk(args.dominance_prediction_dir, topdown=False):
    for name in files:
        if name == "test_results.csv":
            dominance_prediction_files.append(os.path.join(root, name))

print(args.dominance_prediction_dir, dominance_prediction_files)

hi_prediction_files = []
for root, dirs, files in os.walk(args.hi_prediction_dir, topdown=False):
    for name in files:
        if name == "predictions.csv":
            hi_prediction_files.append(os.path.join(root, name))

print(hi_prediction_files)

for dominance_prediction_file in dominance_prediction_files:
    for hi_prediction_file in hi_prediction_files:
        print(dominance_prediction_file.split(args.dominance_prediction_dir)[1])
        d = os.path.split(dominance_prediction_file.split(args.dominance_prediction_dir)[1])[0].strip("/").replace("/", "_")
        h = os.path.split(hi_prediction_file.split(args.hi_prediction_dir)[1])[0].strip("/").replace("/", "_")
        save_dir = os.path.join(args.save_dir, "prob=%s___hi=%s" % (d, h))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Generating results for %s" % save_dir)

        if not os.path.exists(os.path.join(save_dir, "vaccine_score.csv")) or args.force:
            # Generate vaccine scores
            process = subprocess.Popen(['python', 
                'pipeline/1_calc_vaccine_score.py', 
                "--prob_pred_path", 
                dominance_prediction_file,
                "--hi_pred_path",
                hi_prediction_file,
                "--save_path",
                os.path.join(save_dir, "vaccine_score.csv"),
                '--all_sequences_path',
                args.sequence_file
                ],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            print(stdout.decode("utf-8"))
            if stderr:
                print(stderr.decode("utf-8"))
                # exit()

        if not os.path.exists(os.path.join(save_dir, "vaccine_score_and_gt.csv")) or args.force:
            # Generate vaccine scores & ground-truth score
            process = subprocess.Popen(['python', 
                'pipeline/2_calc_gt_vaccine_score.py', 
                "--hi_form_path", 
                args.ground_truth_hi_path,
                "--index_pair",
                os.path.join(save_dir, "vaccine_score.csv"),
                "--ground_truth_path",
                args.ground_truth_prob_path,
                '--sequence_file',
                args.sequence_file,
                ],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            print(stdout.decode("utf-8"))
            if stderr:
                print(stderr.decode("utf-8"))
                # exit()