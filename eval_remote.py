import logging
from evaluator import RemoteLLMConfig, RemoteLLM, run_eval

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.WARN)
    logging.root.setLevel(logging.WARN)
    run_eval(RemoteLLMConfig, RemoteLLM)
