from ....logger import Logger


class ADLogger(Logger):
    ENV_VARIABLE = "AUTO_DEPLOY_LOG_LEVEL"
    PREFIX = "TRT-LLM AUTO-DEPLOY"
    DEFAULT_LEVEL = "info"


ad_logger = ADLogger()
