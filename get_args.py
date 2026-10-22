import argparse
import math
import os

import xlrd
from CLIP.argsbase import get_baseargs

def get_args(main_args):

    # parser = argparse.ArgumentParser()
    parser = get_baseargs()
    # archive_v_DECH,ppo_triplet,ppo_m
    parser.add_argument("--hash-layer", type=str, default="linear", help="choice a hash layer [select, linear] to run. select: select mechaism, linear: sign function.")
    parser.add_argument("--index-file", type=str, default="index.mat")
    # parser.add_argument("--caption-file", type=str, default="caption.mat")
    parser.add_argument("--caption-file", type=str, default="caption.txt")
    parser.add_argument("--label-file", type=str, default="label.mat")

    parser.add_argument("--HM", type=int, default=500)
    # parser.add_argument("--margin", type=int, default=0.1)
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--alpha", type=int, default=1)

    parser.add_argument("--tau", type=int, default=0.1)
    parser.add_argument("--n_bits", type=int, default=32, help="length of hashing binary")

    parser.add_argument("--scaling_p", type=float, default=1.0, help="None")
    parser.add_argument("--scaling_x", type=float, default=3.0, help="None")
    parser.add_argument("--h_dim", type=int, default=0, help="None")
    parser.add_argument("--prior", type=int, default=0, help="None")
    parser.add_argument("--eta", type=float, default=100, help="None")

    parser.add_argument("--eta_e", type=float, default=0.01, help="None")

    parser.add_argument("--smooth_factor", type=float, default=0.15, help="None")
    parser.add_argument("--lam", type=float, default=0.01, help="None")
    parser.add_argument("--m", type=float, default=1, help="None")
    parser.add_argument("--model_lr", type=float, default=0.0001, help="None")
    parser.add_argument("--proxy_nca_lr", type=float, default=0.015, help="None")
    parser.add_argument("--embedding_lr", type=float, default=0.001, help="None")

    parser.add_argument("--trainer", type=str, default="ProgCoPL", help="name of trainer")

    # openset 参数
    parser.add_argument('--fold_idx', type=int, default=1, help='')

    # parser.add_argument("--is-train", action="store_true" ,default=True)


    parser.add_argument('--loss', default='MS', type=str)

    parser.add_argument("--noise-rate", type=float, default=0.0)  # noise 0.2 0.5 0.8

   
    parser.add_argument("--beta_constant", action="store_true")
    parser.add_argument("--k_vals", nargs="+", default=[1, 2, 4, 8], type=int, help="Recall @ Values.")

    parser.add_argument(
        "--lr-beta", default=0.0005, type=float, help="Learning Rate for class margin parameters in MarginLoss"
    )

    parser.add_argument(
        "--n_bins",
        default=50,
        type=int,
        help="Resolution of the support grid, i.e. number of bins between the support interval --bin_limit, K in paper.",
    )

    parser.add_argument(
        "--bin_limit",
        nargs="+",
        default=[0.5, 1.2],
        type=float,
        help="Limit of the sampling distribution support. Excluding small values removes negatives that might be too hard.",
    )

    parser.add_argument(
        "--policy_init_distr",
        default="uniform_low",
        type=str,
        help="Type of initial distribution to use. Default is uniform_low, which places high probabilities between [0.3 and 0.7]. Other options are: random, uniform_high, uniform_avg, uniform_low, uniform_low_and_tight, uniform_lowish_and_tight and the respective normal variants. You may also set uniform/low and set the mean/std in --policy_init_params.",
    )
    parser.add_argument(
        "--policy_init_params",
        nargs="+",
        default=[0.5, 0.04],
        type=float,
        help="Custom initial distribution parameters for either normal or uniform initial distributions: e.g. [mu, sig] = [0.5, 0.04] for normal.",
    )

    parser.add_argument(
        "--policy_merge_oobs",
        action="store_true",
        help="Self-Regularisation Pt.1: Values below lower interval bound are controlled together (same sampling bin).",
    )
    parser.add_argument(
        "--policy_include_pos",
        action="store_true",
        help="Self-Regularisation Pt.2: Include positives into negative sample selection. Excludes positive==anchor.",
    )
    parser.add_argument(
        "--policy_include_same",
        action="store_true",
        help="Self-Regularisation Pt.3: Specifically include positive==anchor in negatives as well.",
    )

    parser.add_argument("--qua", type=float, default=0.1, help="Qua")

    args = parser.parse_args()
    args = argparse.Namespace(**vars(args), **vars(main_args))
    # PPO中修改的参数 ["recall", "map", "dists"]
    # args.policy_merge_oobs = True
    # args.method = 'FTH_crossmodal'
    args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, args.loss, str(args.output_dim))

    return args

# args = get_args()
# sheet = xlrd.open_workbook('./utils/codetable.xlsx').sheet_by_index(0)
# threshold = sheet.row(args.output_dim)[math.ceil(math.log(args.n_classes, 2))].value
