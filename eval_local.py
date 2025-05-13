import logging
from evaluator import run_eval, LocalLLMConfig, LocalLLM

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    run_eval(LocalLLMConfig, LocalLLM)
