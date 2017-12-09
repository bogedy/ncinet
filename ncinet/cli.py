"""
Main interface for ncinet.
"""

from .options import parse_args

def cli():
    options = parse_args()

    if options.train:
        from .train import main
        main(options)
    else:
        from .eval import main
        main(options)
