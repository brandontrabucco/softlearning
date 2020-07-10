import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Plot Evaluation')
    parser.add_argument('--patterns',
                        type=str,
                        nargs='+')
    parser.add_argument('--names',
                        type=str,
                        nargs='+')
    parser.add_argument('--epoch-len',
                        type=int,
                        default=1000)
    parser.add_argument('--tag',
                        type=str,
                        default='ray/tune/evaluation/episode-reward-mean')
    parser.add_argument('--csv-file',
                        type=str,
                        default='data.csv')
    parser.add_argument('--plt-file',
                        type=str,
                        default='data.png')
    parser.add_argument('--title',
                        type=str,
                        default='Evaluation')
    parser.add_argument('--style',
                        type=str,
                        default='darkgrid')
    args = parser.parse_args()

    sns.set(style=args.style)

    df = pd.DataFrame(columns=['Algorithm',
                               'Type',
                               'Timestep',
                               'Average Return'])
    i = 0

    for pattern, name in zip(args.patterns, args.names):

        all_paths = []
        all_path_returns = []

        for t in tf.io.gfile.glob(pattern):

            path = []
            path_return = 0.0

            for e in tf.compat.v1.train.summary_iterator(t):
                for v in e.summary.value:
                    if v.tag == args.tag:
                        path_return += v.simple_value
                        path.append([name,
                                     "All",
                                     e.step * args.epoch_len,
                                     v.simple_value])

                        df.loc[i] = path[-1]
                        i += 1

            all_paths.append(path)
            all_path_returns.append(path_return)

        for e in all_paths[np.argmax(all_path_returns)]:
            df.loc[i] = [e[0], "Max", *e[2:]]
            i += 1

        for e in all_paths[np.argmin(all_path_returns)]:
            df.loc[i] = [e[0], "Min", *e[2:]]
            i += 1

    df.to_csv(args.csv_file)
    sns.lineplot(x="Timestep",
                 y="Average Return",
                 hue="Algorithm",
                 style="Type",
                 data=df)
    plt.title(args.title)
    plt.savefig(args.plt_file)
