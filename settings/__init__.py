from settings.logger_setting import setup_logging
from dotenv import load_dotenv
import getpass
import os

logger = setup_logging()

load_dotenv(verbose=True)

if "LANGSMITH_TRACING" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Please enter your LangSmith tracing config: "
    )

if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Please enter your LangSmith API key: "
    )

if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt="Please enter your LangSmith project name: "
    )

if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass(
        prompt="Please enter your DASHSCOPE API key: "
    )

if "POLYGON_API_KEY" not in os.environ:
    os.environ["POLYGON_API_KEY"] = getpass.getpass(
        prompt="Please enter your Polygon API key: "
    )