import argparse
import glob
import os

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--task", choices=["openbookqa", 'piqa', 'arc_easy', 'arc_challenge', 'sciq'])
    return parser


def main(args):
    files = glob.glob(args.path + "/*/", recursive = True)
    exp = [f.split("/")[-2] for f in files]
    task_exps = []
    for e in exp:
        if args.task in e:
            task_exps.append(e)
    seeds = []
    for e in task_exps:
        seeds.append(e.split("seed")[1].split("_")[0])
    seeds = list(set(seeds))

    results = [{} for s in seeds]
    for s in range(len(seeds)):
        for i in range(len(task_exps)):
            if ("seed" + seeds[s]) in task_exps[i]:
                scores = []
                epochs = []
                try:
                    with open(os.path.join(args.path, task_exps[i] + "/all_results"), "r") as fi:
                        for line in fi:
                            if '"acc": ' in line:
                                scores.append(float(line.strip().split('"acc": ')[1].split(",")[0]))
                            if '"epoch": ' in line:
                                epochs.append(float(line.strip().split('"epoch": ')[1].split(",")[0]))
                    index = 0
                    results[s][task_exps[i]] = [scores[index], epochs[index]]
                except:
                    print(f"Task {task_exps[i]} is not finished")
        sort_results = sorted(results[s].items(), key=lambda x: x[1][0])
        results[s] = sort_results

    for s in range(len(seeds)):
        print("------------------")
        print("seed: ", seeds[s])
        for r in results[s]:
            print(r[0], r[1])


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)