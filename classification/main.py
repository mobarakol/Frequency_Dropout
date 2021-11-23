from arguments import get_args
from solver_cbs import CBSSolver
import os
import random
import numpy as np
import torch

def seed_everything(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = get_args()
    seed_everything(args.seed)

    solver = CBSSolver(args)

    print('training!')
    solver.solve()
    print('done')

    if args.save_model:
        solver.save_model()

if __name__ == '__main__':
    main()
