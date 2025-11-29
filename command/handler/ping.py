import json

from command.registry import command
from utils.logger import get_logger

logger = get_logger("ping")

@command("ping")
async def ping_handler(websocket, state, *args):
    """
    ping command demo
    :param websocket:
    :param state:
    :return:
    """
    logger.info("Ping received, replying pong")
    return {"type": "pong"}